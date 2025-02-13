import os

'''
Changes annotation scheme from IOB to IO.
Adds the PICO annotations created from the EBM model to the Argument Mining annotations, 
so that you don't need to use the EBM model to get annotations live during training and inference. 
Finally, shows some statistics about the PICO annotations in the aggregated files with respect to 
the Argument Relation classes. 
'''

def make_io(file):
    print(f'Converting IOB to IO for file: {file}')
    with open(file, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].strip('\n').split('\t')

    for line in lines:
        if len(line) > 1 and 'B-' in line[-1]:
            tag = line[-1].split('-')[-1]
            line[-1] = 'I-' + tag

    newfile = file.split(".conll")
    newfile = '_io.conll'.join(newfile)
    print(f'Saving to: {newfile}')
    with open(newfile, "w+") as f:
        for line in lines:
            f.write('\t'.join(line) + '\n')

    return True

def aggregate(io_file, pico_file=None):
    newfile = io_file.replace('_for_ner.iopico.conll', '_agg.conll')
    print(f'Aggregating: {io_file} with PICO file: {pico_file}')

    with open(io_file) as f:
        io_lines = [line.strip('\n').split('\t') for line in f.readlines()]

    # Clean extra blank lines
    io_lines = [line for line in io_lines if line != ['']]

    # Handle optional pico_file
    if pico_file and os.path.exists(pico_file):
        with open(pico_file) as f:
            pico_lines = [line.strip('\n').split() for line in f.readlines()]
    else:
        # Fallback to default 'N' if no pico_file
        pico_lines = [['N']] * len(io_lines)

    contents = []
    for i in range(len(io_lines)):
        if len(io_lines[i]) > 1:
            word = io_lines[i][0]
            am_tag = io_lines[i][-1]
            pico_tag = pico_lines[i][1] if len(pico_lines[i]) > 1 else 'N'
            contents.append(f"{word}\t{pico_tag}\t{am_tag}\n")
        else:
            contents.append("\n")

    print(f"Saving aggregated file: {newfile}")
    with open(newfile, 'w') as f:
        f.writelines(contents)

    return newfile

def statistics(agg_file):
    print(f'Drawing statistics for file: {agg_file}')
    with open(agg_file) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].strip('\n').split('\t')

    sentences = []
    sentence = []
    for line in lines:
        if len(line) > 1:
            sentence.append(line)
        else:
            sentences.append(sentence)
            sentence = []

    all_stats = []
    for sentence in sentences:
        am_sentence = set()
        pico_sentence = set()
        for entry in sentence:
            am_sentence.update([entry[-1]])
            pico_sentence.update([entry[-2]])
        all_stats.append([am_sentence, pico_sentence])

    stats = {}
    for sentence in all_stats:
        for am_type in sentence[0]:
            for pico_ent in sentence[1]:
                stats.setdefault(am_type, {'count': 0}).update({pico_ent: stats[am_type].get(pico_ent, 0) + 1})
                stats[am_type]['count'] += 1

    pico_arg_occs = 0
    not_pico_arg_occs = 0
    for sentence in all_stats:
        am_type = sentence[0]
        pico = sentence[1]
        if 'O' not in am_type:
            if pico != {'N'}:
                pico_arg_occs += 1
            else:
                not_pico_arg_occs += 1

    print(f'Arguments with PICO annotations: {pico_arg_occs}')
    print(f'Arguments with no PICO annotations: {not_pico_arg_occs}')

    for am_label in stats:
        print(f'Sentences with class: {am_label}')
        print(f'\t Total occurrences: {stats[am_label]["count"]}')
        for pico_ents in stats[am_label]:
            if pico_ents != 'count':
                percentage = (stats[am_label][pico_ents] / stats[am_label]['count']) * 100
                print(f'\t \t {pico_ents}: {stats[am_label][pico_ents]} ({percentage:.2f}%)')

    return 0

def main():
    for dirpath, dirnames, dirfiles in os.walk('../data/'):
        print(f'Exploring directory: {dirpath}')
        for dirfile in dirfiles:
            if dirfile.endswith('_for_ner.iopico.conll'):
                io_file = os.path.join(dirpath, dirfile)
                pico_file = io_file.replace('_for_ner.iopico.conll', '_for_ner.io.pico.conll')
                agg_file = aggregate(io_file, pico_file)
                if agg_file:
                    print(f"Generated: {agg_file}")

if __name__ == "__main__":
    main()
