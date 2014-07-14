from argparse import ArgumentParser, FileType


def get_args():
    ap = ArgumentParser()
    ap.add_argument('--infile', dest="infile", type=FileType('r'))
    ap.add_argument('--outfile', dest="outfile", type=FileType('w'), metavar="file", default="/tmp/wid_to_class.csv")
    ap.add_argument('--for-secondary', dest="for_secondary", action="store_true", default=False)
    return ap.parse_args()


def primary_transformation(fl):
    results = []
    line_no = 0 # --Isaac
    id_col = 2 # --Isaac
    primary_col = id_col + 1 # --Isaac
    for line in fl:
        splt = line.split(',')
        # following if-clause: on the first iteration, checks 
        # to see if the second cell in the first column is a wikia address (as opposed to  contributor name)
        # This means that all the other fields are shifted over by one.
        # (This is due to an ambiguity of formatting in the google spreadsheets I was given --Isaac)
        if line_no == 1:
            if splt[0][-4:] == '.com':
                id_col = id_col - 1
            primary_col = id_col + 1
        line_no+=1

        results.append(','.join(splt[id_col: primary_col + 1]).lower())
    return "\n".join(results)


def secondary_transformation(fl):
    results = []
    line_no = 0 # --Isaac
    id_col = 2 # --Isaac
    secondary_col = id_col + 2 # --Isaac
    for line in fl:
        splt = line.split(',')
        # following if-clause: on the first iteration, checks 
        # to see if the second cell in the first column is a wikia address (as opposed to  contributor name)
        # This means that all the other fields are shifted over by one.
        # (This is due to an ambiguity of formatting in the google spreadsheets I was given --Isaac)
        if line_no == 1:
            if splt[0][-4:] == '.com':
                id_col = id_col - 1
            primary_col = id_col + 1
        line_no+=1

        [results.append(",".join([splt[id_col], secondary.strip().lower()])) for secondary in splt[min(secondary_col, len(splt)-1)].split('|')]
    return "\n".join(results)


def transform(fl, for_secondary=False):
    return primary_transformation(fl) if not for_secondary else secondary_transformation(fl)


def main():
    args = get_args()
    args.outfile.write(transform(args.infile, args.for_secondary))
    print args.outfile.name



if __name__ == '__main__':
    main()
