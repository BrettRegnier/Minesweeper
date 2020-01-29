from Minesweeper_v1 import Minesweeper_v1
from Broom import Broom


def main():
    ms = Minesweeper_v1(human=False)
    # ms.Play()
    broom = Broom(ms)
    broom.Train(50000)


if __name__ == "__main__":
    main()
