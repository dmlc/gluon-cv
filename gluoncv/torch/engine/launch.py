"""Multiprocessing distributed data parallel support"""
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp


def get_local_ip_and_match(ip_list):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    this_ip = s.getsockname()[0]
    s.close()
    for _i, ip in enumerate(ip_list):
        if ip == this_ip:
            return _i
    return -1


def spawn_workers(main, cfg):
    """Use torch.multiprocessing.spawn to launch distributed processes"""
    if cfg.DDP_CONFIG.AUTO_RANK_MATCH:
        assert len(cfg.DDP_CONFIG.WOLRD_URLS) > 0
        assert cfg.DDP_CONFIG.WOLRD_URLS[0] in cfg.DDP_CONFIG.DIST_URL
        assert len(cfg.DDP_CONFIG.WOLRD_URLS) == cfg.DDP_CONFIG.WORLD_SIZE
        cfg.DDP_CONFIG.WORLD_RANK = get_local_ip_and_match(cfg.DDP_CONFIG.WOLRD_URLS)
        assert cfg.DDP_CONFIG.WORLD_RANK != -1

    ngpus_per_node = torch.cuda.device_count()
    if cfg.DDP_CONFIG.DISTRIBUTED:
        cfg.DDP_CONFIG.GPU_WORLD_SIZE = ngpus_per_node * cfg.DDP_CONFIG.WORLD_SIZE
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, main, cfg))
    else:
        main_worker(cfg.DDP_CONFIG.GPU, ngpus_per_node, main, cfg)


def main_worker(gpu, ngpus_per_node, main, cfg):
    """The main_worker process function (on individual GPU)"""
    cudnn.benchmark = True

    cfg.DDP_CONFIG.GPU = gpu
    print("Use GPU: {}".format(gpu))

    if cfg.DDP_CONFIG.DISTRIBUTED:
        cfg.DDP_CONFIG.GPU_WORLD_RANK = cfg.DDP_CONFIG.WORLD_RANK * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.DDP_CONFIG.DIST_BACKEND,
                                init_method=cfg.DDP_CONFIG.DIST_URL,
                                world_size=cfg.DDP_CONFIG.GPU_WORLD_SIZE,
                                rank=cfg.DDP_CONFIG.GPU_WORLD_RANK)
    main(cfg)
