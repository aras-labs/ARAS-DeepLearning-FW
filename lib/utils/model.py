import lib.backbone as bb
import lib.head as he


def get_model(model_name, args):
    return eval('bb.{}'.format(model_name))(**dict(args))


def get_backbone(model_name, args):
    return eval('bb.{}'.format(model_name))(**dict(args))


def get_head(model_name, args):
    return eval('he.{}'.format(model_name))(**dict(args))
