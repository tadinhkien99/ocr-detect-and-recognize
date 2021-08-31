import os
import sys
import subprocess




import argparse
import os
import sys
from paddle import inference
dirname = os.path.dirname(__file__)

def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=500)

    # params for text detector
    # parser.add_argument("--image_dir", type=str,
    #     default='C:\\Users\\DELL\\Downloads\\AI_FL\\PaddleOCR-release-2.1\\doc\\imgs\\plate_reg.jpg')
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str,
        default=r'./ch_ppocr_mobile_v2.0_det_infer/')
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')

    # DB params
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.6)
    parser.add_argument("--max_batch_size", type=int, default=15)
    parser.add_argument("--use_dilation", type=bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")


    # Rec params
    parser.add_argument("--rec_model_dir", type=str,
        default=r'./en_number_mobile_v2.0_rec_infer/')
    parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
    parser.add_argument("--rec_batch_num", type=int, default=8)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default=os.path.join(dirname, 'en_dict.txt'))
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    # parser.add_argument(
    #     "--vis_font_path", type=str, default=r"C:\Users\DELL\Downloads\AI_FL\PaddleOCR-release-2.1\doc\fonts\simfang.ttf")
    parser.add_argument("--drop_score", type=float, default=0.6)
    return parser.parse_args()


def create_predictor(args, mode):

    if mode == "det":
        # model_dir = args.det_model_dir
        model_file_path = os.path.join(dirname, 'ch_ppocr_mobile_v2.0_det_infer/inference.pdmodel')
        params_file_path = os.path.join(dirname, 'ch_ppocr_mobile_v2.0_det_infer/inference.pdiparams')
    else:
        # model_dir = args.rec_model_dir
        model_file_path = os.path.join(dirname, 'en_number_mobile_v2.0_rec_infer/inference.pdmodel')
        params_file_path = os.path.join(dirname, 'en_number_mobile_v2.0_rec_infer/inference.pdiparams')

    if not os.path.exists(model_file_path):
        print("not find model file path {}".format(model_file_path))
        sys.exit(0)
    if not os.path.exists(params_file_path):
        print("not find params file path {}".format(params_file_path))
        sys.exit(0)

    config = inference.Config(model_file_path, params_file_path)

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
        if args.use_tensorrt:
            config.enable_tensorrt_engine(
                precision_mode=inference.PrecisionType.Half
                if args.use_fp16 else inference.PrecisionType.Float32,
                max_batch_size=args.max_batch_size)
    else:
        config.disable_gpu()
        cpu_threads = args.cpu_threads if hasattr(args, "cpu_threads") else 10
        config.set_cpu_math_library_num_threads(cpu_threads)
    config.enable_memory_optim()
    config.disable_glog_info()

    config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
    config.switch_use_feed_fetch_ops(False)

    # create predictor
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_handle(name)
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_handle(output_name)
        output_tensors.append(output_tensor)
    return predictor, input_tensor, output_tensors

