from fcore.utils.config import parse_config
from fcore.utils.logger import logger
from fcore.utils.distributed import setup_dist_running, get_world_size, is_distributed

def setup():
    # ===========
    # 1. Parse config
    cfg = parse_config()

    # ===========
    # 2. Init distributed running
    setup_dist_running(cfg)

    # ===========
    # 3.Setup logger
    logger.setup_logger(cfg)

    if is_distributed():
        logger.info(f"[!] Distributed Running Initialzed: world size = {get_world_size()}")
    
    return cfg
