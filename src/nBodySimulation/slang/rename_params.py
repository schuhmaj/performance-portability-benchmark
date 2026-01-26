import sys

input_file = sys.argv[1]
output_file = sys.argv[2]
old_name = sys.argv[3]
new_name = sys.argv[4]

with open(input_file, "rb") as f:
    old_text = f.read()

new_text = old_text.replace(old_name.encode('utf-8'), new_name.encode('utf-8'))

with open(output_file, "wb") as f:
    f.write(new_text)