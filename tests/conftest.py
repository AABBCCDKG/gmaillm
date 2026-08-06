import socket

import pytest


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def deny_network(*_args, **_kwargs):
        raise RuntimeError("network access is forbidden in offline tests")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
