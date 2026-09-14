from types import SimpleNamespace

from src.agent import orchestrator


def _tool_call(call_id, name="web_search", arguments='{"query":"faiz"}'):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(*tool_calls, content=None):
    message = SimpleNamespace(tool_calls=list(tool_calls), content=content)
    message.model_dump = lambda exclude_none=True: {
        "role": "assistant",
        **({"content": content} if content is not None else {}),
        **({"tool_calls": list(tool_calls)} if tool_calls else {}),
    }
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(total_tokens=1),
    )


def test_duplicate_tool_call_is_cached_and_loop_is_forced_to_finish(monkeypatch):
    responses = iter([
        _response(_tool_call("call-1", arguments='{"query": "faiz", "max_results": 5}')),
        _response(_tool_call("call-2", arguments='{"max_results":5,"query":"faiz"}')),
        _response(content="Nihai yanıt"),
    ])
    choices = []
    executions = []

    def fake_call_model(client, messages, schemas, tool_choice="auto"):
        choices.append(tool_choice)
        return next(responses)

    monkeypatch.setattr(orchestrator, "get_client", lambda: object())
    monkeypatch.setattr(orchestrator, "_call_model", fake_call_model)
    monkeypatch.setattr(
        orchestrator,
        "execute_tool",
        lambda name, arguments: executions.append((name, arguments)) or '{"results":[]}',
    )

    answer, trace = orchestrator.ask(
        "Bugünkü faiz için web'de ara.", return_trace=True, max_rounds=4
    )

    assert answer == "Nihai yanıt"
    assert len(executions) == 1
    assert [entry["cached"] for entry in trace] == [False, True]
    assert choices == ["auto", "auto", "none"]


def test_different_arguments_keep_chain_progressing(monkeypatch):
    responses = iter([
        _response(_tool_call("call-1", arguments='{"query":"politika faizi"}')),
        _response(_tool_call("call-2", arguments='{"query":"konut kredisi faizi"}')),
        _response(content="İki arama tamamlandı"),
    ])
    executions = []

    monkeypatch.setattr(orchestrator, "get_client", lambda: object())
    monkeypatch.setattr(
        orchestrator,
        "_call_model",
        lambda client, messages, schemas, tool_choice="auto": next(responses),
    )
    monkeypatch.setattr(
        orchestrator,
        "execute_tool",
        lambda name, arguments: executions.append(arguments) or '{"results":[]}',
    )

    answer, trace = orchestrator.ask(
        "Güncel politika ve konut kredisi faizlerini web'de ara.",
        return_trace=True,
        max_rounds=4,
    )

    assert answer == "İki arama tamamlandı"
    assert len(executions) == 2
    assert all(not entry["cached"] for entry in trace)


def test_last_round_is_reserved_for_synthesis(monkeypatch):
    responses = iter([
        _response(_tool_call("call-1")),
        _response(content="Bütçe içinde tamamlandı"),
    ])
    choices = []

    monkeypatch.setattr(orchestrator, "get_client", lambda: object())
    monkeypatch.setattr(orchestrator, "execute_tool", lambda name, arguments: '{}')

    def fake_call_model(client, messages, schemas, tool_choice="auto"):
        choices.append(tool_choice)
        return next(responses)

    monkeypatch.setattr(orchestrator, "_call_model", fake_call_model)

    answer = orchestrator.ask("Bugünkü faiz için web'de ara.", max_rounds=2)

    assert answer == "Bütçe içinde tamamlandı"
    assert choices == ["auto", "none"]
