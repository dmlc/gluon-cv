from trainer import Trainer
from option import Options

def main(args):
    trainer = Trainer(args)
    if args.eval:
        print('Evaluating model: ', args.resume)
        trainer.validation(args.start_epoch)
    elif args.test:
        trainer.test()
    else:
        print('Starting Epoch:', args.start_epoch)
        print('Total Epoches:', args.epochs)
        for epoch in range(args.start_epoch, args.epochs):
            trainer.training(epoch)
            trainer.validation(epoch)


if __name__ == "__main__":
    args = Options().parse()
    main(args)
