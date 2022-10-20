import argparse

def argumentparse():
    
    parse = argparse.ArgumentParser("mydeepfake")
    
    """
    /Root
        /dataset
            /domain_a
            /domain_b
            /voice_a
            /voice_b
            /trans_jp2en
        /result
            /logger
            /model_params
            /output
    """
    # File / Directory 
    parse.add_argument('--root-dir', type=str, default='IORoot', help='Root Directory')
    # Dataset 
    parse.add_argument('--domain-a-dir', type=str, default='domain_a', help='')
    parse.add_argument('--domain-b-dir', type=str, default='domain_b', help='')
    # Result dir
    parse.add_argument('--result-log', type=str, default='result', help='')
    parse.add_argument('--model-params', type=str, default='model_params', help='{model: params, optimizer: params}')
    
    return parse.parse_args()
    

if __name__ == '__main__':
    argumentparse()