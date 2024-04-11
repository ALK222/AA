import argparse


class CommandLine:
    interactive: bool = False
    overfitting: bool = False
    grado: bool = False
    reg: bool = False
    hyperparam: bool = False
    learning_curve: bool = False
    all: bool = False

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Practica 6 - Aprendizaje Automatico')
        self.parser.add_argument('-I', "--Interactive",
                                 help='Interactive mode', required=False, default="", action='store_true')
        self.parser.add_argument('-O', "--Overfitting", help='runs the Overfitting test',
                                 required=False, default="", action='store_true')
        self.parser.add_argument(
            '-G', "--Grado", help='Grado', required=False, default="", action='store_true')
        self.parser.add_argument('-R', "--Regularization", help='runs the Regularization test',
                                 required=False, default="", action='store_true')
        self.parser.add_argument('-HP', "--Hyperparam", help='runs the Hyperparam test',
                                 required=False, default="", action='store_true')
        self.parser.add_argument('-L', "--LearningCurve", help='runs the LearningCurve test',
                                 required=False, default="", action='store_true')
        self.parser.add_argument('-A', "--All", help='runs all tests',
                                 required=False, default="", action='store_true')

    def parse(self, sysargs):
        args = self.parser.parse_args(sysargs)
        if args.Interactive:
            self.interactive = True
        if args.Overfitting:
            self.overfitting = True
        if args.Grado:
            self.grado = True
        if args.Regularization:
            self.reg = True
        if args.Hyperparam:
            self.hyperparam = True
        if args.LearningCurve:
            self.learning_curve = True
        if args.All:
            self.all = True
