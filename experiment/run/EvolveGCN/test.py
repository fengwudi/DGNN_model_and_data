import we_dl as wk
import utils as u

parser = u.create_parser()
args = u.parse_args(parser)

if args.data == 'wikipedia':
    dataset = wk.Wikipedia_Dataset(args)
else:
    dataset = wk.Reddit_Dataset(args)