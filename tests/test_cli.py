from __future__ import annotations

from vocca.cli import main


def test_cli_generate(tmp_path, capsys):
    out = tmp_path / "a.wav"
    code = main(["generate", "你好", "-o", str(out), "--emotion", "happy", "--max-tokens", "16"])
    assert code == 0
    assert out.exists()
    assert "token" in capsys.readouterr().out


def test_cli_generate_stream(tmp_path, capsys):
    out = tmp_path / "s.wav"
    code = main(
        ["generate", "长文本测试", "-o", str(out), "--stream", "--chunk", "4", "--max-tokens", "20"]
    )
    assert code == 0
    assert out.exists()
    assert "流式" in capsys.readouterr().out


def test_cli_generate_with_preset(tmp_path):
    out = tmp_path / "p.wav"
    code = main(
        ["generate", "新闻", "-o", str(out), "--preset", "news_anchor", "--max-tokens", "10"]
    )
    assert code == 0


def test_cli_presets(capsys):
    assert main(["presets"]) == 0
    assert "news_anchor" in capsys.readouterr().out


def test_cli_info(capsys):
    assert main(["info"]) == 0
    assert "vocca" in capsys.readouterr().out


def test_cli_bad_emotion_returns_error(tmp_path, capsys):
    code = main(["generate", "x", "-o", str(tmp_path / "e.wav"), "--emotion", "bogus"])
    assert code == 1
    assert "错误" in capsys.readouterr().err
