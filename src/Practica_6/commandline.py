import argparse


class CommandLine:
    interactive: bool = False

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Practica 6 - Aprendizaje Automatico')
        self.parser.add_argument('-I', "--Interactive",
                                 help='Interactive mode', required=False, default="", action='store_true')

    def parse(self, sysargs):
        args = self.parser.parse_args(sysargs)
        if args.Interactive:
            self.interactive = True
