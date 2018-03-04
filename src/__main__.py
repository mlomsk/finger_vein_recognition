# -*- coding: utf-8 -*-

import argparse

from params import Params


def build_params(args) -> Params:
    import config
    if args.local:
        print("local")
        return config.local
    elif args.devbox:
        print("Devbox")
        return config.devbox


def train_unet(params: Params):
    from nn_models.unet_model import UNetModel
    from data import make_train_generator
    from memory import get_model_memory_usage
    model = UNetModel(params)
    print("Memmory {} GB".format(get_model_memory_usage(params.batch_size, model.model)))
    train_gen, valid_gen = make_train_generator(params)
    model.train(train_gen, valid_gen)


def train_fusion(params: Params):
    from nn_models.fusionnet_model import FusionNetModel
    from data import make_train_generator
    from memory import get_model_memory_usage
    model = FusionNetModel(params)
    print("Memmory {} GB".format(get_model_memory_usage(params.batch_size, model.model)))
    train_gen, valid_gen = make_train_generator(params)
    model.train(train_gen, valid_gen)


def predict(params: Params):
    from keras.models import load_model
    from submission import make_submission
    from metrics import mean_iou, dice_coef_loss, dice_coef
    model = load_model(params.model_path,
                       {'mean_iou': mean_iou, 'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})
    make_submission(model, params)


def main():
    parser = argparse.ArgumentParser(prog="Bowl 2018")

    env_group = parser.add_mutually_exclusive_group(required=True)
    env_group.add_argument("--local", action='store_true')
    env_group.add_argument("--devbox", action='store_true')

    train_group = parser.add_argument_group("train")
    train_group.add_argument("--train", action='store_true')
    train_group.add_argument("--model_name", type=str)

    pred_group = parser.add_argument_group("prediction")
    pred_group.add_argument("--predict", action='store_true')
    pred_group.add_argument("--model_path", type=str)

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--depth_mask", action='store_true')

    args = parser.parse_args()
    params = build_params(args)
    train = {'unet': train_unet, 'fusionnet': train_fusion}
    if args.train:
        params.setup_train(args.model_name)
        train[args.model_name](params)
    elif args.predict:
        params.setup_submission()
        params.model_path = args.model_path
        predict(params)
    else:
        print("Nothing to do.")


if __name__ == '__main__':
    main()
