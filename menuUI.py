import pygame
import sys

# UI Menu for choosing spawn rate
def show_menu():
    global chosen_spawn_rate  # Ensure chosen_spawn_rate is global
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("Choose Spawn Rate")

    font = pygame.font.Font(None, 36)
    white, black, gray = (255, 255, 255), (0, 0, 0), (200, 200, 200)

    buttons = {
        "Low": (200, 100, 200, 50),
        "Medium": (200, 170, 200, 50),
        "High": (200, 240, 200, 50)
    }

    while True:
        screen.fill(white)
        title = font.render("Choose Vehicle Spawn Rate", True, black)
        screen.blit(title, (screen.get_width() // 2 - title.get_width() // 2, 30))

        for text, rect in buttons.items():
            pygame.draw.rect(screen, gray, rect)
            label = font.render(text, True, black)
            screen.blit(label, (rect[0] + 50, rect[1] + 10))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                for text, rect in buttons.items():
                    if rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]:
                        chosen_spawn_rate = {"Low": 3, "Medium": 1, "High": 0.5}[text]
                        return chosen_spawn_rate  # Return the selected spawn rate

