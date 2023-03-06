import logging, os, os.path as osp


def get_logger(name, work_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    if not work_dir:
        work_dir = osp.join('work_dirs', name)
    os.makedirs(work_dir, exist_ok=True)
    console = logging.StreamHandler()  # 控制台句柄
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    handler = logging.FileHandler(osp.join(work_dir, 'train.log'))  # 文件句柄
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(f"{name}Log创建完成")
    return logger, work_dir


if __name__ == "__main__":
    logger,_ = get_logger('nothing')
    for i in range(1000):
        logger.info("woaini")
