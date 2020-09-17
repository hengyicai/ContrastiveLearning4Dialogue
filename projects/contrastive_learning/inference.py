#: coding: utf-8

import os
import locale

locale.setlocale(locale.LC_ALL, 'en_US')
from parlai.scripts.eval_model import setup_args
from parlai.core.agents import create_agent
from projects.robust_dialog.utils import create_batch_from_file


def inference(opt, batch_arr, out_file):
    model = create_agent(opt, requireModelExists=True)

    with open(out_file, 'w') as f:
        for batch in batch_arr:
            out = model.inference(batch)
            text = out.text
            for t in text:
                to_print = t
                f.write(to_print)
                f.write('\n')


def main():
    parser = setup_args()
    parser.add_argument('--inference_input_file', type=str, required=True)
    parser.add_argument('--inference_batchsize', type=int, default=1)
    parser.add_argument('--inference_output_file', type=str, default='')
    parser.add_argument('--suffix', type=str, default='predictions')
    opt = parser.parse_args()

    batch_size = opt['inference_batchsize']
    input_context_file = opt['inference_input_file']
    if opt['inference_output_file'] == '':
        assert opt['suffix'] != '', "You have not specify the <inference_output_file>, so the <suffix> must be given!"
        output_file = "{}.{}".format(input_context_file, opt['suffix'])
    else:
        output_dir = os.path.dirname(opt['inference_output_file'])
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        output_file = opt['inference_output_file']
    batch_arr = create_batch_from_file(input_context_file, batch_size=batch_size)

    inference(opt, batch_arr, output_file)


if __name__ == '__main__':
    main()
