import os
import argparse
import json

from annoy import AnnoyIndex


def get_arguments():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(help='sub-commands', dest='task')

    build_tree_parser = sub_parsers.add_parser(
        'build', help='Create a forest of trees')
    build_tree_parser.add_argument(
        '--id_name',  help='id name', required=True)
    build_tree_parser.add_argument(
        '--dim',  help='length of vector', type=int, required=True)
    build_tree_parser.add_argument(
        '--treeFile', help='tree file path', required=True)
    build_tree_parser.add_argument(
        '--idxFile', help='index file path', required=True)
    build_tree_parser.add_argument(
        '--ntree', help='More ntree gives higher precision', type=int, required=True)
    build_tree_parser.add_argument(
        '--input', help='workflow start yyyymmddhh', required=True)

    nn_parser = sub_parsers.add_parser('nn', help='Get closeest items')
    nn_parser.add_argument(
        '--id_name',  help='id name', required=True)
    nn_parser.add_argument(
        '--dim',  help='length of vector', type=int, required=True)
    nn_parser.add_argument(
        '--treeFile', help='tree file path', required=True)
    nn_parser.add_argument(
        '--idxFile', help='index file path', required=True)
    nn_parser.add_argument(
        '--output', help='output file path', required=True)

    all_parser = sub_parsers.add_parser(
        'all', help='Create a forest of trees and Get cloesest items')
    all_parser.add_argument(
        '--id_name',  help='id name', required=True)
    all_parser.add_argument(
        '--dim',  help='length of vector', type=int, required=True)
    all_parser.add_argument(
        '--treeFile', help='tree file path', required=True)
    all_parser.add_argument(
        '--idxFile', help='index file path', required=True)
    all_parser.add_argument(
        '--ntree', help='More ntree gives higher precision', type=int, required=True)
    all_parser.add_argument(
        '--input', help='workflow start yyyymmddhh', required=True)
    all_parser.add_argument(
        '--output', help='output file path', required=True)


    return parser.parse_args()

class Ann:
    @staticmethod
    def build(id_name, dim, input, ntree, treeFile, idxFile):
        print(f'build {id_name} tree')

        if os.path.exists(treeFile):
            os.remove(treeFile)
        
        if os.path.exists(idxFile):
            os.remove(idxFile)

        with open(input, 'r') as ifs:
            annoyIndex = AnnoyIndex(dim, 'angular')
            idx2id = {}
            idx = 0
            for line in ifs:
                if idx % 1000 == 0:
                    print(f'Read: {int(idx / 1000)}K')

                pair = json.loads(line)
                
                id = pair[id_name]
                vec = pair['vector']
                annoyIndex.add_item(idx, vec)
                idx2id[idx] = id
                idx += 1

            print(f'>> Read Total {idx} items')

        annoyIndex.build(n_trees=ntree, n_jobs=-1)
        annoyIndex.save(treeFile)

        with open(idxFile, 'a') as ofs:
            for idx in idx2id:
                pair = {
                    'idx': idx,
                    id_name: idx2id[idx]
                }

                line = json.dumps(pair)
                ofs.write(line + '\n')

    @staticmethod
    def nn(id_name, dim, treeFile, idxFile, output):
        print(f'nn search {id_name}')
        if os.path.exists(output):
            os.remove(output)

        idx2id = {}
        with open(idxFile, 'r') as ifs:
            for line in ifs:
                pair = json.loads(line)
                idx2id[pair['idx']] = pair[id_name]

        annoyIndex = AnnoyIndex(dim, 'angular')
        annoyIndex.load(treeFile)

        with open(output, 'a') as ofs:
            for idx in idx2id:
                nn_indices = annoyIndex.get_nns_by_item(idx, 10)

                neghbors = []

                for nn_idx in nn_indices:
                    id = idx2id[nn_idx]
                    neghbors.append(id)

                item = {
                    id_name: idx2id[idx],
                    'nn': neghbors
                }

                line = json.dumps(item)

                ofs.write(line + '\n')


def main():
    args = get_arguments()

    if args.task == 'build':
        print(f'>> id_name: {args.id_name} ')
        print(f'>> dim: {args.dim} ')
        print(f'>> input: {args.input} ')
        print(f'>> ntree: {args.ntree} ')
        print(f'>> treeFile: {args.treeFile} ')
        print(f'>> idxFile: {args.idxFile} ')

        Ann.build(id_name=args.id_name, dim=args.dim, input=args.input,
                  ntree=args.ntree, treeFile=args.treeFile, idxFile=args.idxFile)
    elif args.task == 'nn':
        print(f'>> id_name: {args.id_name} ')
        print(f'>> dim: {args.dim} ')
        print(f'>> output: {args.output} ')
        print(f'>> treeFile: {args.treeFile} ')
        print(f'>> idxFile: {args.idxFile} ')
        Ann.nn(id_name=args.id_name, dim=args.dim, output=args.output,
               treeFile=args.treeFile, idxFile=args.idxFile)
    elif args.task == 'all':
        print(f'>> id_name: {args.id_name} ')
        print(f'>> dim: {args.dim} ')
        print(f'>> input: {args.input} ')
        print(f'>> output: {args.output} ')
        print(f'>> treeFile: {args.treeFile} ')
        print(f'>> idxFile: {args.idxFile} ')
        Ann.build(id_name=args.id_name, dim=args.dim, input=args.input,
                  ntree=args.ntree, treeFile=args.treeFile, idxFile=args.idxFile)
        Ann.nn(id_name=args.id_name, dim=args.dim, output=args.output,
               treeFile=args.treeFile, idxFile=args.idxFile)

    else:
        raise Exception(f"Invalid task: {args.task}")


if __name__ == '__main__':
    main()
