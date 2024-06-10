import pygame
import random
import math

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Simulation parameters
WIDTH = 800
HEIGHT = 800
FPS = 20
ROBOT_SIZE = 30
SENSOR_LENGTH = 500
NUM_LIDAR_SENSORS = 1  # Number of LiDAR sensors around the robot
SENSOR_ANGLE_OFFSET = 360 / NUM_LIDAR_SENSORS
MOVEMENT_SPEED = 2
TURN_ANGLE = 5  # Turn angle during rotation
NUM_PARTICLES = 100  # Number of particles in the particle filter

# Obstacles
OBSTACLES = [
    pygame.Rect(250, 250, 100, 100),  # Example obstacle
    pygame.Rect(600, 600, 150, 50)    # Example obstacle
]

def draw_obstacles(screen):
    for obstacle in OBSTACLES:
        pygame.draw.rect(screen, BLACK, obstacle)

def is_collision(robot_rect):
    for obstacle in OBSTACLES:
        if robot_rect.colliderect(obstacle):
            return True
    return False

def draw_robot(screen, pos, angle):
    robot_surface = pygame.Surface((ROBOT_SIZE, ROBOT_SIZE), pygame.SRCALPHA)
    pygame.draw.circle(robot_surface, RED, (ROBOT_SIZE // 2, ROBOT_SIZE // 2), ROBOT_SIZE // 2)
    rotated_robot = pygame.transform.rotate(robot_surface, -angle)
    new_rect = rotated_robot.get_rect(center=(pos[0], pos[1]))
    screen.blit(rotated_robot, new_rect.topleft)

def draw_lidar(screen, robot_pos, angle):
    for i in range(NUM_LIDAR_SENSORS):
        sensor_angle = angle + i * SENSOR_ANGLE_OFFSET
        sensor_end_x = robot_pos[0] + math.cos(math.radians(sensor_angle)) * SENSOR_LENGTH
        sensor_end_y = robot_pos[1] - math.sin(math.radians(sensor_angle)) * SENSOR_LENGTH
        pygame.draw.line(screen, BLACK, robot_pos, (sensor_end_x, sensor_end_y), 1)

def sensor(robot_pos, angle):
    sensor_length = 0

    while sensor_length < SENSOR_LENGTH:
        sensor_x = robot_pos[0] + sensor_length * math.cos(math.radians(angle))
        sensor_y = robot_pos[1] - sensor_length * math.sin(math.radians(angle))

        if not (0 <= sensor_x < WIDTH and 0 <= sensor_y < HEIGHT):
            return sensor_length  # Return the distance to the boundary

        for obstacle in OBSTACLES:
            if obstacle.collidepoint(sensor_x, sensor_y):
                return sensor_length  # Return the distance to the obstacle

        sensor_length += 1

    return SENSOR_LENGTH  # Return the max sensor length if no obstacle or boundary is found

def lidar_sensors(robot_pos, robot_angle):
    distances = []
    for i in range(NUM_LIDAR_SENSORS):
        sensor_angle = (robot_angle + i * SENSOR_ANGLE_OFFSET) % 360
        distances.append(sensor(robot_pos, sensor_angle))
    return distances

def draw_particles(screen, particles):
    for particle in particles:
        pygame.draw.circle(screen, BLACK, (int(particle[0]), int(particle[1])), 2)
        end_x = particle[0] + 10 * math.cos(math.radians(particle[2]))
        end_y = particle[1] - 10 * math.sin(math.radians(particle[2]))
        pygame.draw.line(screen, BLACK, (int(particle[0]), int(particle[1])), (int(end_x), int(end_y)), 1)

def initialize_particles(num_particles, width, height):
    particles = []
    for _ in range(num_particles):
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        angle = random.uniform(0, 360)
        particles.append([x, y, angle, 1.0 / num_particles])
    return particles

def move_particles(particles, movement_speed, turn_angle, width, height):
    for particle in particles:
        angle_rad = math.radians(particle[2])
        particle[0] += movement_speed * math.cos(angle_rad) + random.gauss(0, 1)
        particle[1] -= movement_speed * math.sin(angle_rad) + random.gauss(0, 1)
        particle[2] = (particle[2] + turn_angle + random.gauss(0, 1)) % 360

        # Ensure particles remain within bounds
        particle[0] = min(max(particle[0], 0), width)
        particle[1] = min(max(particle[1], 0), height)

def weight_particles(particles, robot_distances):
    weights = []
    for particle in particles:
        particle_distances = lidar_sensors((particle[0], particle[1]), particle[2])
        print(particle_distances)
        weight = 1.0
        for pd, rd in zip(particle_distances, robot_distances):
            weight *= math.exp(-((pd - rd) ** 2) / 100.0)  # Waga Gaussa. 100 służy do skalowania wyników. Kwadrat różnnicy dystansu zanegowany i podzielony przez 100, a następnie wartość wykładnicza e^x
        particle[3] = weight
        weights.append(weight)
    return weights
def normalize_weights(weights):
    total = sum(weights)
    if total == 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]

def resample_particles(particles, weights):
    new_particles = []
    n = len(particles)
    indices = sorted(range(n), key=lambda i: weights[i])
    index = int(random.random() * n)
    beta = 0.0
    for _ in range(n):
        beta += random.random() * max(weights)
        while beta > weights[index]:
            beta -= weights[index]
            index = (index + 1) % n
        new_particles.append(particles[index][:])

        # Zmiana cząstki z pewnym prawdopodobieństwem
        if random.random() < beta / max(weights):
            # Losowa zmiana pozycji cząstki
            # new_particles[-1][0] += random.uniform(-1000, 1000)
            # new_particles[-1][1] += random.uniform(-1000, 1000)
            # new_particles[-1][2] += random.uniform(-1000, 1000)
            new_particles[-1][0] += random.uniform(-10, 10)
            new_particles[-1][1] += random.uniform(-10, 10)
            new_particles[-1][2] += random.uniform(-10, 10)

    return new_particles
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Particle Filter with LiDAR")
    clock = pygame.time.Clock()
    robot_pos = [WIDTH / 2, HEIGHT / 2]
    robot_angle = 0

    particles = initialize_particles(NUM_PARTICLES, WIDTH, HEIGHT)

    running = True
    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            new_x = robot_pos[0] + MOVEMENT_SPEED * math.cos(math.radians(robot_angle))
            new_y = robot_pos[1] - MOVEMENT_SPEED * math.sin(math.radians(robot_angle))
            new_rect = pygame.Rect(new_x - ROBOT_SIZE / 2, new_y - ROBOT_SIZE / 2, ROBOT_SIZE, ROBOT_SIZE)
            if not is_collision(new_rect) and 0 <= new_x <= WIDTH and 0 <= new_y <= HEIGHT:
                robot_pos = [new_x, new_y]

        if keys[pygame.K_LEFT]:
            robot_angle += TURN_ANGLE
        if keys[pygame.K_RIGHT]:
            robot_angle -= TURN_ANGLE

        robot_angle = robot_angle % 360

        # Move particles based on the same control inputs
        move_particles(particles, MOVEMENT_SPEED if keys[pygame.K_UP] else 0,
                       TURN_ANGLE if keys[pygame.K_LEFT] else (-TURN_ANGLE if keys[pygame.K_RIGHT] else 0),
                       WIDTH, HEIGHT)

        draw_obstacles(screen)
        draw_robot(screen, robot_pos, robot_angle)
        draw_lidar(screen, robot_pos, robot_angle)
        draw_particles(screen, particles)

        robot_distances = lidar_sensors(robot_pos, robot_angle)
        print(robot_distances)

        # Weight particles based on sensor readings
        weights = weight_particles(particles, robot_distances)
        weights = normalize_weights(weights)
        # Resample particles based on their weights
        particles = resample_particles(particles, weights)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()