import Minesweeper_v0
import Broom

def main():
    ms = Minesweeper_v0.Minesweeper_v0()
    # ms.Play()
    broom = Broom.Broom(ms)
    broom.Train(50000)
        
if __name__ == "__main__":
    main()
    