from milvus_client.common.args import build_common_parser


def test_common_parser_accepts_protocol_options(tmp_path):
    parser = build_common_parser("test")
    args = parser.parse_args(
        [
            "--uri",
            "http://localhost:19530",
            "--token",
            "root:Milvus",
            "--collection-prefix",
            "qa",
            "--duration-sec",
            "60",
            "--seed",
            "123",
            "--feature-set",
            "compat_2_6",
            "--compat-mode",
            "rollback_safe",
            "--capability-probe",
            "true",
            "--skip-unsupported",
            "true",
            "--lifecycle-phase",
            "steady_state",
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--output-json",
            str(tmp_path / "result.json"),
        ]
    )

    assert args.uri == "http://localhost:19530"
    assert args.token == "root:Milvus"
    assert args.collection_prefix == "qa"
    assert args.duration_sec == 60
    assert args.seed == 123
    assert args.feature_set == "compat_2_6"
    assert args.compat_mode == "rollback_safe"
    assert args.capability_probe is True
    assert args.skip_unsupported is True
    assert args.lifecycle_phase == "steady_state"
