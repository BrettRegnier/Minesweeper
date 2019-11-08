import Minesweeper_v0
import Broom

def main():
    ms = Minesweeper_v0.Minesweeper_v0()
    # while True:
    #     ms.Update()
    #     ms.Render()
    broom = Broom.Broom(ms)
    broom.Train(20000)
        
if __name__ == "__main__":
    main()
    