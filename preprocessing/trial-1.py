# Function to add blank lines after each full stop
def add_blank_lines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        previous_line_empty = False  # To handle the issue of extra blank lines
        for line in infile:
            # Split each line by spaces/tabs
            batch = line.strip().split()

            # If the line is not empty and has 4 columns
            if batch:
                # Write the current line
                outfile.write(line)
                
                # Check if the token (batch[1]) is a full stop and ensure it's not already followed by a blank line
                if len(batch) > 1 and batch[1] == '.' and not previous_line_empty:
                    outfile.write('\n')  # Add a blank line after the sentence
                    previous_line_empty = True  # Remember that a blank line was added
                else:
                    previous_line_empty = False  # Reset if no full stop

# Example usage
input_file = '../data/neoplasm/train_agg.conll'  # The input CoNLL file
output_file = 'train_agg_with_blank_lines.conll'  # Output file with added blank lines

add_blank_lines(input_file, output_file)