import os
from crop_animal import crop
from build_profile import build_profile

def main():
    os.makedirs("data", exist_ok=True) # create data folder (other files dependant on it)
    crop() # Crops animal photos
    build_profile() # build embedding profile for animal

if __name__ == "__main__":
    main()