try:
    from common_legacy import *  # type: ignore # noqa: F401,F403
except ImportError:
    from milvus_client.common_legacy import *  # type: ignore # noqa: F401,F403
