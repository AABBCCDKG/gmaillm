import base64
import importlib.metadata
import importlib.util
import json
import os
import stat
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gmail_newsletter_summarizer as main


class FakeRequest:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def execute(self):
        if self.error is not None:
            raise self.error
        return self.result


class FakeMessages:
    def __init__(self, list_results=None, get_result=None, send_result=None):
        self.list_results = list(list_results or [])
        self.get_result = get_result
        self.send_result = send_result
        self.list_calls = []
        self.get_calls = []
        self.send_calls = []

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        return FakeRequest(result=self.list_results.pop(0))

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        return FakeRequest(result=self.get_result)

    def send(self, **kwargs):
        self.send_calls.append(kwargs)
        if isinstance(self.send_result, Exception):
            return FakeRequest(error=self.send_result)
        return FakeRequest(result=self.send_result)


class FakeGmail:
    def __init__(self, messages):
        self._messages = messages

    def users(self):
        return self

    def messages(self):
        return self._messages


def encode_body(value):
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


class TestQueries:
    def test_build_query_uses_one_valid_sender_before_dates(self):
        result = main.build_query(
            ["newsletter@example.com"],
            "2026/08/01",
            "2026/08/05",
        )

        assert result == (
            "from:newsletter@example.com "
            "after:2026/08/01 before:2026/08/05"
        )

    @pytest.mark.parametrize(
        "sender",
        [
            "safe@example.com OR newer:1d",
            "safe@example.com}",
            "from:attacker@example.com",
            '"safe@example.com"',
            "Display Name <safe@example.com>",
            "safe*wildcard@example.com",
            "safe@example.com\nOR from:attacker@example.com",
        ],
    )
    def test_build_query_rejects_gmail_query_injection(self, sender):
        with pytest.raises(ValueError, match="valid email"):
            main.build_query([sender], "2026/08/01", None)

    def test_build_query_rejects_multiple_senders(self):
        with pytest.raises(ValueError, match="Exactly one sender"):
            main.build_query(
                ["first@example.com", "second@example.com"],
                "2026/08/01",
                None,
            )

    @pytest.mark.parametrize("senders", [None, [], ["", "  "]])
    def test_build_query_requires_one_sender(self, senders):
        with pytest.raises(ValueError, match="Exactly one sender"):
            main.build_query(senders, "2026/08/01", None)

    @pytest.mark.parametrize(
        ("date_after", "date_before"),
        [
            ("2026/08/01 newer:1d", None),
            ("2026-08-01", None),
            ("2026/02/30", None),
            ("2026/08/01", "2026/08/05 OR from:attacker@example.com"),
        ],
    )
    def test_build_query_rejects_date_query_injection(
        self, date_after, date_before
    ):
        with pytest.raises(ValueError, match="YYYY/MM/DD"):
            main.build_query(
                ["news@example.com"],
                date_after,
                date_before,
            )

    def test_build_query_uses_configured_timezone_for_default_date(self):
        with mock.patch.object(main, "get_current_date", return_value="2026/08/06") as date:
            result = main.build_query(
                ["news@example.com"],
                None,
                None,
                timezone_str="Asia/Tokyo",
            )

        assert result == "from:news@example.com after:2026/08/06"
        date.assert_called_once_with(timezone_str="Asia/Tokyo")

    def test_fetch_emails_reads_every_page(self):
        messages = FakeMessages(
            list_results=[
                {"messages": [{"id": "one"}], "nextPageToken": "next"},
                {"messages": [{"id": "two"}]},
            ]
        )

        result = main.fetch_emails(
            FakeGmail(messages),
            email_filter_list=["news@example.com"],
            date_after_filter="2026/08/01",
        )

        assert result == [{"id": "one"}, {"id": "two"}]
        assert messages.list_calls[0] == {
            "userId": "me",
            "q": "from:news@example.com after:2026/08/01",
        }
        assert messages.list_calls[1] == {
            "userId": "me",
            "q": "from:news@example.com after:2026/08/01",
            "pageToken": "next",
        }

    def test_fetch_emails_wraps_api_failure(self):
        class BrokenMessages(FakeMessages):
            def list(self, **kwargs):
                return FakeRequest(error=OSError("offline"))

        with pytest.raises(RuntimeError, match="Failed to fetch emails"):
            main.fetch_emails(
                FakeGmail(BrokenMessages()),
                email_filter_list=["news@example.com"],
                date_after_filter="2026/08/01",
            )


class TestParsing:
    def test_parse_email_body_reads_root_text_plain_payload(self):
        message = {
            "payload": {
                "mimeType": "text/plain",
                "body": {"data": encode_body("root body")},
            }
        }

        assert main.parse_email_body(message) == "root body"

    def test_parse_email_body_recurses_into_nested_multipart(self):
        message = {
            "payload": {
                "mimeType": "multipart/mixed",
                "parts": [
                    {
                        "mimeType": "multipart/alternative",
                        "parts": [
                            {
                                "mimeType": "text/html",
                                "body": {"data": encode_body("<p>html</p>")},
                            },
                            {
                                "mimeType": "text/plain",
                                "body": {"data": encode_body("nested body")},
                            },
                        ],
                    }
                ],
            }
        }

        assert main.parse_email_body(message) == "nested body"

    def test_parse_email_body_rejects_invalid_base64(self):
        message = {
            "payload": {
                "mimeType": "text/plain",
                "body": {"data": "%%%"},
            }
        }

        with pytest.raises(ValueError, match="base64"):
            main.parse_email_body(message)

    def test_parse_email_data_handles_header_names_case_insensitively(self):
        messages = FakeMessages(
            get_result={
                "payload": {
                    "headers": [
                        {"name": "subject", "value": "Daily update"},
                        {"name": "FROM", "value": "News <news@example.com>"},
                    ],
                    "mimeType": "text/plain",
                    "body": {"data": encode_body("Body")},
                }
            }
        )

        result = main.parse_email_data(FakeGmail(messages), {"id": "message-id"})

        assert result == {
            "subject": "Daily update",
            "sender": "News <news@example.com>",
            "body": "Body",
        }

    def test_parse_email_data_rejects_missing_sender(self):
        messages = FakeMessages(
            get_result={
                "payload": {
                    "headers": [{"name": "Subject", "value": "No sender"}],
                    "mimeType": "text/plain",
                    "body": {"data": encode_body("Body")},
                }
            }
        )

        with pytest.raises(ValueError, match="From"):
            main.parse_email_data(FakeGmail(messages), {"id": "message-id"})


class TestSummaries:
    def test_create_openai_completion_uses_legacy_chat_completion_api(self):
        if importlib.util.find_spec("openai") is None:
            pytest.skip("openai is not installed in this interpreter")
        import openai

        assert importlib.metadata.version("openai") == "0.27.8"
        with mock.patch.object(
            openai.ChatCompletion,
            "create",
            return_value={"choices": [{"message": {"content": '{"General": []}'}}]},
        ) as create:
            result = main.create_openai_completion(
                api_key="test-key",
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.2,
                response_format={"type": "json_object"},
            )

        assert result["choices"][0]["message"]["content"] == '{"General": []}'
        create.assert_called_once_with(
            api_key="test-key",
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

    def test_summarize_email_retries_then_returns_valid_json(self):
        attempts = []

        def completion_create(**kwargs):
            attempts.append(kwargs)
            if len(attempts) < 3:
                raise TimeoutError("temporary")
            return {"choices": [{"message": {"content": '{"Technology": []}'}}]}

        result = main.summarize_email(
            [{"body": "Newsletter", "sender": "A", "subject": "B"}],
            completion_create=completion_create,
            token_counter=lambda _text, _model: 10,
            max_attempts=3,
        )

        assert result == '{"Technology": []}'
        assert len(attempts) == 3

    def test_summarize_email_rejects_over_limit_before_api_call(self):
        completion_create = mock.Mock()

        with pytest.raises(ValueError, match="Token length"):
            main.summarize_email(
                [{"body": "Newsletter", "sender": "A", "subject": "B"}],
                completion_create=completion_create,
                token_counter=lambda _text, _model: 101,
                token_length=100,
            )

        completion_create.assert_not_called()

    def test_summarize_email_rejects_empty_email_collection(self):
        with pytest.raises(ValueError, match="No email bodies"):
            main.summarize_email(
                [],
                completion_create=mock.Mock(),
                token_counter=lambda _text, _model: 0,
            )

    def test_summarize_email_retries_invalid_json_response(self):
        responses = iter(
            [
                {"choices": [{"message": {"content": "not-json"}}]},
                {"choices": [{"message": {"content": '{"General": []}'}}]},
            ]
        )

        result = main.summarize_email(
            [{"body": "Newsletter", "sender": "A", "subject": "B"}],
            completion_create=lambda **_kwargs: next(responses),
            token_counter=lambda _text, _model: 10,
            max_attempts=2,
        )

        assert json.loads(result) == {"General": []}

    def test_summarize_email_raises_after_retry_budget_is_exhausted(self):
        completion_create = mock.Mock(side_effect=TimeoutError("offline"))

        with pytest.raises(RuntimeError, match="after 2 attempts"):
            main.summarize_email(
                [{"body": "Newsletter", "sender": "A", "subject": "B"}],
                completion_create=completion_create,
                token_counter=lambda _text, _model: 10,
                max_attempts=2,
                retry_delay_seconds=0,
            )

        assert completion_create.call_count == 2


class TestPreview:
    def test_get_senders_preserves_each_email_from_the_same_sender(self):
        email_data = [
            {
                "sender": "Daily News <news@example.com>",
                "subject": "Morning Brief",
                "body": "first private body",
            },
            {
                "sender": "Daily News <news@example.com>",
                "subject": "Evening Brief",
                "body": "second private body",
            },
        ]

        assert main.get_senders(email_data) == [
            ("Daily News", "Morning Brief"),
            ("Daily News", "Evening Brief"),
        ]

    def test_main_previews_every_email_and_count_without_bodies_before_send(self):
        config = main.AppConfig(
            openai_api_key="key",
            recipient=None,
            sender_filters=("news@example.com",),
            model="model",
            token_length_limit=100,
            timezone="UTC",
            token_file=Path("token.json"),
            credentials_file=Path("credentials.json"),
            summary_dir=Path("summary"),
        )
        email_data = [
            {
                "sender": "Daily News <news@example.com>",
                "subject": "Morning Brief",
                "body": "first private body",
            },
            {
                "sender": "Daily News <news@example.com>",
                "subject": "Evening Brief",
                "body": "second private body",
            },
        ]
        events = []

        def input_func(prompt):
            events.append(("prompt", prompt))
            return "STOP"

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main, "load_environment"))
            stack.enter_context(mock.patch.object(main, "load_config", return_value=config))
            stack.enter_context(
                mock.patch.object(main, "get_gmail_client", return_value=object())
            )
            stack.enter_context(
                mock.patch.object(
                    main,
                    "fetch_emails",
                    return_value=[{"id": "one"}, {"id": "two"}],
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main,
                    "parse_email_data",
                    side_effect=email_data,
                )
            )
            summarize = stack.enter_context(
                mock.patch.object(main, "summarize_email")
            )
            result = main.main(
                input_func=input_func,
                output_func=lambda line: events.append(("output", line)),
            )

        assert result == 0
        assert events[:5] == [
            ("output", "The following emails will be summarized:"),
            ("output", "Total emails: 2"),
            ("output", "- Daily News: Morning Brief"),
            ("output", "- Daily News: Evening Brief"),
            (
                "prompt",
                "Selected newsletter bodies will be sent to OpenAI for processing. "
                "Type SEND to continue: ",
            ),
        ]
        rendered_preview = "\n".join(value for _kind, value in events)
        assert "first private body" not in rendered_preview
        assert "second private body" not in rendered_preview
        summarize.assert_not_called()


class TestOutput:
    def test_convert_to_markdown_rejects_non_object_json(self):
        with pytest.raises(ValueError, match="JSON object"):
            main.convert_to_markdown("[]")

    def test_convert_to_markdown_drops_unsafe_urls(self):
        result = main.convert_to_markdown(
            json.dumps(
                {
                    "General": [
                        {
                            "summary": "Do not run scripts",
                            "url": "javascript:alert(1)",
                        }
                    ]
                }
            )
        )

        assert result == "### General\n\n- Do not run scripts\n"

    def test_convert_to_markdown_escapes_generated_html(self):
        result = main.convert_to_markdown(
            json.dumps(
                {
                    "<script>category</script>": [
                        {
                            "summary": "<img src=x onerror=alert(1)>",
                            "url": "https://example.com",
                        }
                    ]
                }
            )
        )

        assert "<script>" not in result
        assert "<img" not in result
        assert "&lt;script&gt;" in result
        assert r"&lt;img src=x onerror=alert\(1\)&gt;" in result

    def test_convert_to_markdown_escapes_link_breakout_text(self):
        result = main.convert_to_markdown(
            json.dumps(
                {
                    "General": [
                        {
                            "summary": "safe](javascript:alert(1))",
                            "url": "https://example.com/article_(one)",
                        }
                    ]
                }
            )
        )

        assert r"safe\]\(javascript:alert\(1\)\)" in result
        assert "https://example.com/article_%28one%29" in result
        assert "](javascript:" not in result

    def test_send_email_returns_api_response(self):
        messages = FakeMessages(send_result={"id": "sent-id"})
        fake_markdown = SimpleNamespace(markdown=lambda value: f"<p>{value}</p>")

        with mock.patch.dict(sys.modules, {"markdown": fake_markdown}):
            result = main.send_email(
                FakeGmail(messages),
                "recipient@example.com",
                "Subject",
                "Body",
            )

        assert result == {"id": "sent-id"}
        assert messages.send_calls[0]["userId"] == "me"
        assert messages.send_calls[0]["body"]["raw"]

    def test_send_email_raises_on_api_failure(self):
        messages = FakeMessages(send_result=OSError("offline"))
        fake_markdown = SimpleNamespace(markdown=lambda value: f"<p>{value}</p>")

        with mock.patch.dict(sys.modules, {"markdown": fake_markdown}):
            with pytest.raises(RuntimeError, match="Failed to send email"):
                main.send_email(
                    FakeGmail(messages),
                    "recipient@example.com",
                    "Subject",
                    "Body",
                )


class TestConfiguration:
    def test_load_config_uses_one_sender_without_hardcoded_recipient(self):
        env = {
            "OPENAI_API_KEY": "key",
            "GMAIL_SENDER_FILTERS": "News@Example.COM",
            "OPENAI_MODEL": "model-name",
        }

        with mock.patch.dict(os.environ, env, clear=True):
            config = main.load_config()

        assert config.recipient is None
        assert config.sender_filters == ("News@example.com",)
        assert config.openai_api_key == "key"
        assert config.model == "model-name"

    @pytest.mark.parametrize(
        "sender_value",
        [
            "one@example.com,two@example.com",
            "one@example.com OR from:two@example.com",
            "Display Name <one@example.com>",
            "one@example",
            "one@@example.com",
        ],
    )
    def test_load_config_rejects_multiple_or_unsafe_senders(self, sender_value):
        with mock.patch.dict(
            os.environ,
            {"GMAIL_SENDER_FILTERS": sender_value},
            clear=True,
        ):
            with pytest.raises(ValueError, match="sender|email"):
                main.load_config()

    def test_load_config_rejects_unknown_timezone(self):
        with mock.patch.dict(
            os.environ,
            {
                "GMAIL_SENDER_FILTERS": "newsletter@example.com",
                "GMAIL_LLM_TIMEZONE": "Not/A-Timezone",
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="Unknown GMAIL_LLM_TIMEZONE"):
                main.load_config()

    def test_load_config_requires_explicit_sender_allowlist(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GMAIL_SENDER_FILTERS"):
                main.load_config()

    def test_load_config_resolves_relative_paths_from_project_root(self):
        with mock.patch.dict(
            os.environ,
            {
                "GMAIL_SENDER_FILTERS": "newsletter@example.com",
                "GMAIL_TOKEN_FILE": "private/token.json",
                "GMAIL_CREDENTIALS_FILE": "private/credentials.json",
                "GMAIL_LLM_SUMMARY_DIR": "output",
            },
            clear=True,
        ):
            config = main.load_config()

        assert config.token_file == main.PROJECT_ROOT / "private/token.json"
        assert config.credentials_file == main.PROJECT_ROOT / "private/credentials.json"
        assert config.summary_dir == main.PROJECT_ROOT / "output"

    def test_load_config_accepts_legacy_project_variable_names(self):
        with mock.patch.dict(
            os.environ,
            {
                "GMAIL_SENDER_FILTERS": "newsletter@example.com",
                "GMAILLM_TIMEZONE": "UTC",
                "GMAILLM_SUMMARY_DIR": "legacy-output",
            },
            clear=True,
        ):
            config = main.load_config()

        assert config.timezone == "UTC"
        assert config.summary_dir == main.PROJECT_ROOT / "legacy-output"

    def test_main_reports_send_failure_without_false_success(self):
        config = main.AppConfig(
            openai_api_key="key",
            recipient="recipient@example.com",
            sender_filters=(),
            model="model",
            token_length_limit=100,
            timezone="UTC",
            token_file=Path("token.json"),
            credentials_file=Path("credentials.json"),
            summary_dir=Path("summary"),
        )
        output = []
        inputs = iter(["SEND", "email"])

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main, "load_environment"))
            stack.enter_context(mock.patch.object(main, "load_config", return_value=config))
            stack.enter_context(
                mock.patch.object(main, "get_gmail_client", return_value=object())
            )
            stack.enter_context(
                mock.patch.object(main, "fetch_emails", return_value=[{"id": "1"}])
            )
            stack.enter_context(
                mock.patch.object(
                    main,
                    "parse_email_data",
                    return_value={"sender": "A", "subject": "B", "body": "Body"},
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main, "summarize_email", return_value='{"General": []}'
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main, "send_email", side_effect=RuntimeError("offline")
                )
            )
            result = main.main(
                input_func=lambda _prompt: next(inputs),
                output_func=output.append,
            )

        assert result == 1
        assert any("Failed to send" in line for line in output)
        assert not any("sent via email" in line for line in output)

    def test_main_passes_configured_timezone_to_gmail_query(self):
        config = main.AppConfig(
            openai_api_key="key",
            recipient=None,
            sender_filters=("newsletter@example.com",),
            model="model",
            token_length_limit=100,
            timezone="Asia/Tokyo",
            token_file=Path("token.json"),
            credentials_file=Path("credentials.json"),
            summary_dir=Path("summary"),
        )

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main, "load_environment"))
            stack.enter_context(mock.patch.object(main, "load_config", return_value=config))
            stack.enter_context(
                mock.patch.object(main, "get_gmail_client", return_value=object())
            )
            fetch = stack.enter_context(
                mock.patch.object(main, "fetch_emails", return_value=[])
            )
            result = main.main(output_func=lambda _line: None)

        assert result == 0
        fetch.assert_called_once_with(
            mock.ANY,
            email_filter_list=["newsletter@example.com"],
            timezone_str="Asia/Tokyo",
        )

    def test_main_skips_unreadable_messages(self):
        config = main.AppConfig(
            openai_api_key="key",
            recipient="recipient@example.com",
            sender_filters=(),
            model="model",
            token_length_limit=100,
            timezone="UTC",
            token_file=Path("token.json"),
            credentials_file=Path("credentials.json"),
            summary_dir=Path("summary"),
        )
        output = []
        inputs = iter(["SEND", "email"])

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main, "load_environment"))
            stack.enter_context(mock.patch.object(main, "load_config", return_value=config))
            stack.enter_context(
                mock.patch.object(main, "get_gmail_client", return_value=object())
            )
            stack.enter_context(
                mock.patch.object(
                    main,
                    "fetch_emails",
                    return_value=[{"id": "bad"}, {"id": "good"}],
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main,
                    "parse_email_data",
                    side_effect=[
                        ValueError("missing From"),
                        {"sender": "A", "subject": "B", "body": "Body"},
                    ],
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main, "summarize_email", return_value='{"General": []}'
                )
            )
            stack.enter_context(
                mock.patch.object(main, "send_email", return_value={"id": "sent"})
            )
            result = main.main(
                input_func=lambda _prompt: next(inputs),
                output_func=output.append,
            )

        assert result == 0
        assert any("Skipping an unreadable email" in line for line in output)

    def test_private_token_write_is_atomic_and_owner_only(self, tmp_path):
        token_file = tmp_path / "private" / "token.json"

        main._write_private_token(token_file, '{"refresh_token": "synthetic"}')

        assert token_file.read_text(encoding="utf-8") == (
            '{"refresh_token": "synthetic"}'
        )
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
        assert stat.S_IMODE(token_file.parent.stat().st_mode) == 0o700

    def test_existing_token_permissions_are_repaired(self, tmp_path):
        token_file = tmp_path / "token.json"
        token_file.write_text("synthetic", encoding="utf-8")
        token_file.chmod(0o644)

        main._restrict_private_file(token_file)

        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600

    def test_existing_token_symlink_is_rejected(self, tmp_path):
        real_token = tmp_path / "real-token.json"
        real_token.write_text("synthetic", encoding="utf-8")
        token_file = tmp_path / "token.json"
        token_file.symlink_to(real_token)

        with pytest.raises(OSError, match="symlink"):
            main._restrict_private_file(token_file)

        assert token_file.is_symlink()
        assert real_token.read_text(encoding="utf-8") == "synthetic"

    def test_summary_write_repairs_directory_and_file_permissions(self, tmp_path):
        summary_dir = tmp_path / "summary"
        summary_dir.mkdir(mode=0o755)
        summary_file = summary_dir / "2026-08-06.md"
        summary_file.write_text("old", encoding="utf-8")
        summary_file.chmod(0o644)

        main.write_private_summary(summary_file, "new summary")

        assert summary_file.read_text(encoding="utf-8") == "new summary"
        assert stat.S_IMODE(summary_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(summary_file.stat().st_mode) == 0o600
        assert list(summary_dir.glob(f".{summary_file.name}.*")) == []

    def test_summary_write_rejects_symlink_directory(self, tmp_path):
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        summary_dir = tmp_path / "summary"
        summary_dir.symlink_to(real_dir, target_is_directory=True)

        with pytest.raises(OSError, match="symlink"):
            main.write_private_summary(summary_dir / "summary.md", "private")


class TestGmailAuthorization:
    def test_invalid_token_cache_is_removed_with_reauthorization_message(
        self, tmp_path
    ):
        from google.auth.exceptions import GoogleAuthError
        from google.oauth2.credentials import Credentials

        token_file = tmp_path / "token.json"
        token_file.write_text("not-json", encoding="utf-8")

        with mock.patch.object(
            Credentials,
            "from_authorized_user_file",
            side_effect=GoogleAuthError("invalid cache"),
        ):
            with pytest.raises(
                main.GmailAuthorizationError,
                match="removed.*authorize Gmail",
            ):
                main.get_gmail_client(
                    token_file=token_file,
                    credentials_file=tmp_path / "credentials.json",
                )

        assert not token_file.exists()

    def test_missing_credentials_file_has_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="credentials file not found"):
            main.get_gmail_client(
                token_file=tmp_path / "token.json",
                credentials_file=tmp_path / "credentials.json",
            )

    def test_first_authorization_writes_private_cache_and_builds_client(
        self, tmp_path
    ):
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient import discovery

        credentials_file = tmp_path / "credentials.json"
        credentials_file.write_text("{}", encoding="utf-8")
        token_file = tmp_path / "private" / "token.json"
        credentials = mock.Mock()
        credentials.to_json.return_value = '{"token": "synthetic"}'
        flow = mock.Mock()
        flow.run_local_server.return_value = credentials
        client = object()

        with mock.patch.object(
            InstalledAppFlow,
            "from_client_secrets_file",
            return_value=flow,
        ) as create_flow:
            with mock.patch.object(discovery, "build", return_value=client) as build:
                result = main.get_gmail_client(token_file, credentials_file)

        assert result is client
        create_flow.assert_called_once_with(str(credentials_file), main.SCOPES)
        flow.run_local_server.assert_called_once_with(port=0)
        build.assert_called_once_with(
            "gmail",
            "v1",
            credentials=credentials,
            cache_discovery=False,
        )
        assert token_file.read_text(encoding="utf-8") == '{"token": "synthetic"}'
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600

    def test_browser_authorization_error_has_retry_guidance(self, tmp_path):
        from google.auth.exceptions import GoogleAuthError
        from google_auth_oauthlib.flow import InstalledAppFlow

        credentials_file = tmp_path / "credentials.json"
        credentials_file.write_text("{}", encoding="utf-8")
        flow = mock.Mock()
        flow.run_local_server.side_effect = GoogleAuthError("denied")

        with mock.patch.object(
            InstalledAppFlow,
            "from_client_secrets_file",
            return_value=flow,
        ):
            with pytest.raises(
                main.GmailAuthorizationError,
                match="complete the browser authorization flow",
            ):
                main.get_gmail_client(
                    token_file=tmp_path / "token.json",
                    credentials_file=credentials_file,
                )

    def test_client_build_auth_error_removes_cached_token(self, tmp_path):
        from google.auth.exceptions import GoogleAuthError
        from google.oauth2.credentials import Credentials
        from googleapiclient import discovery

        token_file = tmp_path / "token.json"
        token_file.write_text("synthetic", encoding="utf-8")
        credentials = mock.Mock(valid=True)

        with mock.patch.object(
            Credentials,
            "from_authorized_user_file",
            return_value=credentials,
        ):
            with mock.patch.object(
                discovery,
                "build",
                side_effect=GoogleAuthError("rejected"),
            ):
                with pytest.raises(
                    main.GmailAuthorizationError,
                    match="token cache was removed",
                ):
                    main.get_gmail_client(
                        token_file=token_file,
                        credentials_file=tmp_path / "credentials.json",
                    )

        assert not token_file.exists()

    def test_discard_invalid_token_unlinks_symlink_only(self, tmp_path):
        real_token = tmp_path / "real-token.json"
        real_token.write_text("synthetic", encoding="utf-8")
        token_file = tmp_path / "token.json"
        token_file.symlink_to(real_token)

        main._discard_invalid_token(token_file)

        assert not token_file.exists()
        assert real_token.read_text(encoding="utf-8") == "synthetic"


class TestAdditionalValidation:
    def test_load_environment_reads_project_files_in_override_order(self):
        fake_dotenv = SimpleNamespace(load_dotenv=mock.Mock())

        with mock.patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            main.load_environment()

        assert fake_dotenv.load_dotenv.call_args_list == [
            mock.call(main.PROJECT_ROOT / ".env"),
            mock.call(main.PROJECT_ROOT / ".env.local", override=True),
        ]

    @pytest.mark.parametrize("token_limit", ["invalid", "0", "-1"])
    def test_load_config_rejects_invalid_token_limit(self, token_limit):
        with mock.patch.dict(
            os.environ,
            {
                "GMAIL_SENDER_FILTERS": "newsletter@example.com",
                "OPENAI_TOKEN_LIMIT": token_limit,
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="OPENAI_TOKEN_LIMIT"):
                main.load_config()

    def test_get_current_date_returns_specific_date(self):
        assert main.get_current_date("2026/08/06", "UTC") == "2026/08/06"

    def test_parse_email_body_without_data_is_empty(self):
        assert (
            main.parse_email_body(
                {"payload": {"mimeType": "text/plain", "body": {}}}
            )
            == ""
        )

    def test_parse_email_body_rejects_missing_payload(self):
        with pytest.raises(ValueError, match="payload"):
            main.parse_email_body({})

    def test_parse_email_data_rejects_missing_id(self):
        with pytest.raises(ValueError, match="missing an id"):
            main.parse_email_data(object(), {})

    def test_parse_email_data_wraps_message_failure(self):
        messages = FakeMessages(get_result=None)

        def broken_get(**_kwargs):
            return FakeRequest(error=OSError("offline"))

        messages.get = broken_get
        with pytest.raises(RuntimeError, match="Failed to load email"):
            main.parse_email_data(FakeGmail(messages), {"id": "message-id"})

    def test_parse_email_data_rejects_missing_payload(self):
        messages = FakeMessages(get_result={})

        with pytest.raises(ValueError, match="has no payload"):
            main.parse_email_data(FakeGmail(messages), {"id": "message-id"})

    @pytest.mark.parametrize("emails", ["body", b"body", object()])
    def test_summarize_email_rejects_non_sequence_mapping_collection(self, emails):
        with pytest.raises(TypeError, match="sequence"):
            main.summarize_email(emails)

    def test_summarize_email_rejects_zero_attempts(self):
        with pytest.raises(ValueError, match="at least one"):
            main.summarize_email(
                [{"body": "Newsletter"}],
                completion_create=mock.Mock(),
                token_counter=lambda _text, _model: 1,
                max_attempts=0,
            )

    def test_summarize_email_requires_api_key_for_default_client(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            main.summarize_email(
                [{"body": "Newsletter"}],
                token_counter=lambda _text, _model: 1,
            )

    def test_summarize_email_rejects_non_object_response(self):
        with pytest.raises(RuntimeError, match="after 1 attempts"):
            main.summarize_email(
                [{"body": "Newsletter"}],
                completion_create=lambda **_kwargs: {
                    "choices": [{"message": {"content": "[]"}}]
                },
                token_counter=lambda _text, _model: 1,
                max_attempts=1,
            )

    def test_completion_content_accepts_object_response(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"A": []}'))]
        )

        assert main._completion_content(response) == '{"A": []}'

    @pytest.mark.parametrize(
        "response",
        [
            {},
            {"choices": []},
            {"choices": [{"message": {"content": ""}}]},
        ],
    )
    def test_completion_content_rejects_missing_or_empty_content(self, response):
        with pytest.raises(ValueError, match="content"):
            main._completion_content(response)

    def test_convert_to_markdown_rejects_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            main.convert_to_markdown("not-json")

    def test_convert_to_markdown_rejects_non_list_category(self):
        with pytest.raises(ValueError, match="must contain a list"):
            main.convert_to_markdown('{"General": {}}')

    def test_convert_to_markdown_rejects_non_mapping_item(self):
        with pytest.raises(ValueError, match="invalid item"):
            main.convert_to_markdown('{"General": ["bad"]}')

    def test_send_email_rejects_blank_recipient(self):
        with pytest.raises(ValueError, match="must not be blank"):
            main.send_email(object(), " ", "Subject", "Body")


class TestMainOutcomes:
    @staticmethod
    def config(tmp_path, *, api_key="key", recipient="recipient@example.com"):
        return main.AppConfig(
            openai_api_key=api_key,
            recipient=recipient,
            sender_filters=("newsletter@example.com",),
            model="model",
            token_length_limit=100,
            timezone="UTC",
            token_file=tmp_path / "token.json",
            credentials_file=tmp_path / "credentials.json",
            summary_dir=tmp_path / "summary",
        )

    def test_main_writes_and_emails_summary_successfully(self, tmp_path):
        config = self.config(tmp_path)
        inputs = iter(["SEND", "both"])
        output = []

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main, "load_environment"))
            stack.enter_context(mock.patch.object(main, "load_config", return_value=config))
            stack.enter_context(
                mock.patch.object(main, "get_gmail_client", return_value=object())
            )
            stack.enter_context(
                mock.patch.object(main, "fetch_emails", return_value=[{"id": "1"}])
            )
            stack.enter_context(
                mock.patch.object(
                    main,
                    "parse_email_data",
                    return_value={
                        "sender": "News <newsletter@example.com>",
                        "subject": "Daily",
                        "body": "Body",
                    },
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main,
                    "summarize_email",
                    return_value=(
                        '{"General": [{"summary": "Item", '
                        '"url": "https://example.com"}]}'
                    ),
                )
            )
            send = stack.enter_context(
                mock.patch.object(main, "send_email", return_value={"id": "sent"})
            )
            stack.enter_context(
                mock.patch.object(main, "get_current_date", return_value="2026/08/06")
            )
            result = main.main(
                input_func=lambda _prompt: next(inputs),
                output_func=output.append,
            )

        summary_file = config.summary_dir / "2026-08-06.md"
        assert result == 0
        assert stat.S_IMODE(config.summary_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(summary_file.stat().st_mode) == 0o600
        assert "[Item](https://example.com)" in summary_file.read_text(
            encoding="utf-8"
        )
        send.assert_called_once()
        assert any("Summary saved" in line for line in output)
        assert "Summary sent via email." in output

    def test_main_reports_configuration_failure(self):
        output = []

        with mock.patch.object(
            main,
            "load_environment",
            side_effect=ValueError("bad config"),
        ):
            result = main.main(output_func=output.append)

        assert result == 2
        assert output == ["Configuration error: bad config"]

    def test_main_requires_openai_api_key(self, tmp_path):
        config = self.config(tmp_path, api_key=None)
        output = []

        with mock.patch.object(main, "load_environment"):
            with mock.patch.object(main, "load_config", return_value=config):
                result = main.main(output_func=output.append)

        assert result == 2
        assert output == ["Configuration error: OPENAI_API_KEY is required."]

    def test_main_reports_gmail_access_failure(self, tmp_path):
        config = self.config(tmp_path)
        output = []

        with mock.patch.object(main, "load_environment"):
            with mock.patch.object(main, "load_config", return_value=config):
                with mock.patch.object(
                    main,
                    "get_gmail_client",
                    side_effect=main.GmailAuthorizationError("reauthorize"),
                ):
                    result = main.main(output_func=output.append)

        assert result == 1
        assert output == ["Failed to access Gmail: reauthorize"]

    def test_main_reports_summary_failure(self, tmp_path):
        config = self.config(tmp_path)
        output = []

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main, "load_environment"))
            stack.enter_context(mock.patch.object(main, "load_config", return_value=config))
            stack.enter_context(
                mock.patch.object(main, "get_gmail_client", return_value=object())
            )
            stack.enter_context(
                mock.patch.object(main, "fetch_emails", return_value=[{"id": "1"}])
            )
            stack.enter_context(
                mock.patch.object(
                    main,
                    "parse_email_data",
                    return_value={"sender": "A", "subject": "B", "body": "Body"},
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main,
                    "summarize_email",
                    side_effect=RuntimeError("OpenAI offline"),
                )
            )
            result = main.main(
                input_func=lambda _prompt: "SEND",
                output_func=output.append,
            )

        assert result == 1
        assert any("Failed to summarize emails" in line for line in output)

    def test_main_rejects_invalid_output_choice(self, tmp_path):
        config = self.config(tmp_path)
        inputs = iter(["SEND", "print"])

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main, "load_environment"))
            stack.enter_context(mock.patch.object(main, "load_config", return_value=config))
            stack.enter_context(
                mock.patch.object(main, "get_gmail_client", return_value=object())
            )
            stack.enter_context(
                mock.patch.object(main, "fetch_emails", return_value=[{"id": "1"}])
            )
            stack.enter_context(
                mock.patch.object(
                    main,
                    "parse_email_data",
                    return_value={"sender": "A", "subject": "B", "body": "Body"},
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main,
                    "summarize_email",
                    return_value='{"General": []}',
                )
            )
            result = main.main(
                input_func=lambda _prompt: next(inputs),
                output_func=lambda _line: None,
            )

        assert result == 2

    def test_main_requires_recipient_for_email_output(self, tmp_path):
        config = self.config(tmp_path, recipient=None)
        inputs = iter(["SEND", "email"])

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main, "load_environment"))
            stack.enter_context(mock.patch.object(main, "load_config", return_value=config))
            stack.enter_context(
                mock.patch.object(main, "get_gmail_client", return_value=object())
            )
            stack.enter_context(
                mock.patch.object(main, "fetch_emails", return_value=[{"id": "1"}])
            )
            stack.enter_context(
                mock.patch.object(
                    main,
                    "parse_email_data",
                    return_value={"sender": "A", "subject": "B", "body": "Body"},
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main,
                    "summarize_email",
                    return_value='{"General": []}',
                )
            )
            stack.enter_context(
                mock.patch.object(main, "get_current_date", return_value="2026/08/06")
            )
            result = main.main(
                input_func=lambda _prompt: next(inputs),
                output_func=lambda _line: None,
            )

        assert result == 2

    def test_main_reports_summary_write_failure(self, tmp_path):
        config = self.config(tmp_path)
        inputs = iter(["SEND", "file"])

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main, "load_environment"))
            stack.enter_context(mock.patch.object(main, "load_config", return_value=config))
            stack.enter_context(
                mock.patch.object(main, "get_gmail_client", return_value=object())
            )
            stack.enter_context(
                mock.patch.object(main, "fetch_emails", return_value=[{"id": "1"}])
            )
            stack.enter_context(
                mock.patch.object(
                    main,
                    "parse_email_data",
                    return_value={"sender": "A", "subject": "B", "body": "Body"},
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main,
                    "summarize_email",
                    return_value='{"General": []}',
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main,
                    "write_private_summary",
                    side_effect=OSError("read-only"),
                )
            )
            stack.enter_context(
                mock.patch.object(main, "get_current_date", return_value="2026/08/06")
            )
            result = main.main(
                input_func=lambda _prompt: next(inputs),
                output_func=lambda _line: None,
            )

        assert result == 1

    def test_refresh_error_removes_expired_token_cache(self, tmp_path):
        from google.auth.exceptions import RefreshError
        from google.oauth2.credentials import Credentials

        token_file = tmp_path / "token.json"
        token_file.write_text("synthetic", encoding="utf-8")
        credentials = mock.Mock(
            valid=False,
            expired=True,
            refresh_token="refresh-token",
        )
        credentials.refresh.side_effect = RefreshError("revoked")

        with mock.patch.object(
            Credentials,
            "from_authorized_user_file",
            return_value=credentials,
        ):
            with pytest.raises(
                main.GmailAuthorizationError,
                match="expired or was revoked.*authorize Gmail",
            ):
                main.get_gmail_client(
                    token_file=token_file,
                    credentials_file=tmp_path / "credentials.json",
                )

        assert not token_file.exists()
