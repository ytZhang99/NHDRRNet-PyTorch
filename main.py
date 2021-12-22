from train import Trainer
from option import args


def main():
    if args.test_only:
        t = Trainer()
        t.test(args.ep)
    else:
        t = Trainer()
        t.train()


if __name__ == '__main__':
    main()
=======
from train import Trainer
from option import args


def main():
    if args.test_only:
        t = Trainer()
        t.test(args.ep)
    else:
        t = Trainer()
        t.train()


if __name__ == '__main__':
    main()
>>>>>>> 320cf2305e588f0eba31d550eec35c4282346f15
