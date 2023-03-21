class Writer:
    def add_scalars(self, tag_scalar_dic, global_step):
        raise NotImplementedError()

    def add_scalars_with_prefix(self, tag_scalar_dic, global_step, prefix):
        tag_scalar_dic = {prefix + k: v for k, v in tag_scalar_dic.items()}
        self.add_scalars(tag_scalar_dic, global_step)


class TBWriter(Writer):
    def __init__(self, dir_path):
        from tensorboardX import SummaryWriter

        self.writer = SummaryWriter(dir_path, flush_secs=30)

    def add_scalars(self, tag_scalar_dic, global_step):
        for tag, scalar in tag_scalar_dic.items():
            self.writer.add_scalar(tag, scalar, global_step)


def get_writer(dir_path):
    """
    Args:
        dir_path: tb dir
    """
    writer = TBWriter(dir_path)

    return writer
