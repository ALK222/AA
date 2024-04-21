import argparse


class CommandLine:
    interactive: bool = False
    plot: bool = False
    complex: bool = False
    simple: bool = False
    regularized: bool = False
    iter: bool = False
    all: bool = False

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Practica 6 - Aprendizaje Automatico')
        self.parser.add_argument('-I', "--Interactive",
                                 help='Interactive mode', required=False, default="", action='store_true')
        self.parser.add_argument('-P', "--Plot", help='plots the data',
                                 required=False, default="", action='store_true')
        self.parser.add_argument('-C', "--Complex", help='runs complex model',
                                 required=False, default="", action='store_true')
        self.parser.add_argument('-S', "--Simple", help='runs simple model',
                                 required=False, default="", action='store_true')
        self.parser.add_argument('-R', "--Regularized", help='runs regularized model',
                                 required=False, default="", action='store_true')
        self.parser.add_argument('-It', "--Iter", help='runs iter model',
                                 required=False, default="", action='store_true')
        self.parser.add_argument('-A', "--All", help='runs all tests',
                                 required=False, default="", action='store_true')

    def parse(self, sysargs):
        args = self.parser.parse_args(sysargs)
        if args.Interactive:
            self.interactive = True
        if args.Plot:
            self.plot = True
        if args.Complex:
            self.complex = True
        if args.Simple:
            self.simple = True
        if args.Regularized:
            self.regularized = True
        if args.Iter:
            self.iter = True
        if args.All:
            self.all = True
