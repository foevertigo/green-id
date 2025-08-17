from planting.video_processing.verify_video import verify_video_planting_from_url
from planting.geo_verification.check_gps import get_gps_coords
from planting.utils.download_media import download_from_url
import os

def verify_geo_match(img_url_1, img_url_2):
    print("Comparing GPS between two images")
    img1_path = "planting/uploads/weekly_images/img1.jpg"
    img2_path = "planting/uploads/weekly_images/img2.jpg"
    download_from_url(img_url_1, img1_path)
    download_from_url(img_url_2, img2_path)

    gps1 = get_gps_coords(img1_path)
    gps2 = get_gps_coords(img2_path)

    if not gps1 or not gps2:
        print("One or both images lack GPS metadata.")
        return False

    result = abs(gps1[0] - gps2[0]) < 0.001 and abs(gps1[1] - gps2[1]) < 0.001
    print("GPS Match" if result else "GPS Mismatch")
    return result

# Example usage
if __name__ == "__main__":
    video_url = "https://your-cdn.com/videos/week0.mp4"
    week0_img_url = "https://your-cdn.com/images/week0.jpg"
    week1_img_url = "https://your-cdn.com/images/week1.jpg"

    print("Starting Planting Verification Pipeline")
    planting_valid = verify_video_planting_from_url(video_url)
    geo_valid = verify_geo_match(week0_img_url, week1_img_url)

    if planting_valid and geo_valid:
        print("Final Result: Verified Planting Activity")
    else:
        print("Final Result: Verification Failed")

