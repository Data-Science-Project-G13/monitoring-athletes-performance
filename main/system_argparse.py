import os
import argparse

def parse_options(internal_args: []):
    parser = argparse.ArgumentParser(description='Run System')
    parser.add_argument('--initialize-system', dest='do_initialize', required=False,
                        type=lambda x: (str(x).lower() == 'true'),
                        default=True, help='whether initialize the system')
    parser.add_argument('--clean-data', dest='do_cleaning', required=False,
                        type=lambda x: (str(x).lower() == 'true'),
                        default=True, help='whether clean the athlete data')
    parser.add_argument('--process-feature-engineering', dest='do_feature_engineering', required=False,
                        type=lambda x: (str(x).lower() == 'true'),
                        default=True, help='whether process feature engineering')
    parser.add_argument('--build-model', dest='do_modeling', required=False,
                        type=lambda x: (str(x).lower() == 'true'),
                        default=True, help='whether process modeling')
    parser.add_argument('--generate-pmc', dest='do_pmc_generating', required=False,
                        type=lambda x: (str(x).lower() == 'true'),
                        default=True, help='whether generate pmc')
    parser.add_argument('--athletes-names', nargs='+', dest='athletes_names', required=True,
                        help='whether generate pmc')
    if internal_args:
        options = vars(parser.parse_args(internal_args))
        options['athletes_names'] = list(filter(lambda x:
                                                '--athletes-names' in x, internal_args))[0].split('=')[1].split()
    else:
        options = vars(parser.parse_args())
    return options


if __name__ == '__main__':
    # print(parse_options([]))  # For testing the way of using command-line/Terminal
    print(parse_options(["--initialize-system=False", '--clean-data=False', "--athletes-names=xu_chen eduardo_oliveira carly_hart"]))
