from optparse import OptionParser
opts = OptionParser()
opts.add_option("--parameters", "-d", help = "...")   # python classExample.py -d 1 2 3
options, args = opts.parse_args()
print(options, "\n", args)