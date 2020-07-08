from PIL import Image
import os
import json
import numpy as np
import random
from collections import defaultdict

N_CLASSES = 200

def mask_image(file_path, out_dir_name, remove_bkgnd=True):
    """
    Remove background or foreground using segmentation label
    """
    im = np.array(Image.open(file_path).convert('RGB'))
    segment_path = file_path.replace('images', 'segmentations').replace('.jpg', '.png')
    segment_im = np.array(Image.open(segment_path).convert('L'))
    #segment_im = np.tile(segment_im, (3,1,1)) #3 x W x H
    #segment_im = np.moveaxis(segment_im, 0, -1) #W x H x 3
    mask = segment_im.astype(float)/255
    if not remove_bkgnd: #remove bird in the foreground instead
        mask = 1 - mask
    new_im = (im * mask[:, :, None]).astype(np.uint8)
    Image.fromarray(new_im).save(file_path.replace('/images/', out_dir_name))

def mask_dataset(test_pkl, out_dir_name, remove_bkgnd=True):
    data = pickle.load(open(test_pkl, 'rb'))
    file_paths = [d['img_path'] for d in data]
    for file_path in file_paths:
        mask_image(file_path, out_dir_name, remove_bkgnd)

def crop_and_resize(source_img, target_img):
    """
    Make source_img exactly the same as target_img by expanding/shrinking and
    cropping appropriately.

    If source_img's dimensions are strictly greater than or equal to the
    corresponding target img dimensions, we crop left/right or top/bottom
    depending on aspect ratio, then shrink down.

    If any of source img's dimensions are smaller than target img's dimensions,
    we expand the source img and then crop accordingly

    Modified from
    https://stackoverflow.com/questions/4744372/reducing-the-width-height-of-an-image-to-fit-a-given-aspect-ratio-how-python
    """
    source_width = source_img.size[0]
    source_height = source_img.size[1]

    target_width = target_img.size[0]
    target_height = target_img.size[1]

    # Check if source does not completely cover target
    if (source_width < target_width) or (source_height < target_height):
        # Try matching width
        width_resize = (target_width, int((target_width / source_width) * source_height))
        if (width_resize[0] >= target_width) and (width_resize[1] >= target_height):
            source_resized = source_img.resize(width_resize, Image.ANTIALIAS)
        else:
            height_resize = (int((target_height / source_height) * source_width), target_height)
            assert (height_resize[0] >= target_width) and (height_resize[1] >= target_height)
            source_resized = source_img.resize(height_resize, Image.ANTIALIAS)
        # Rerun the cropping
        return crop_and_resize(source_resized, target_img)

    source_aspect = source_width / source_height
    target_aspect = target_width / target_height

    if source_aspect > target_aspect:
        # Crop left/right
        new_source_width = int(target_aspect * source_height)
        offset = (source_width - new_source_width) // 2
        resize = (offset, 0, source_width - offset, source_height)
    else:
        # Crop top/bottom
        new_source_height = int(source_width / target_aspect)
        offset = (source_height - new_source_height) // 2
        resize = (0, offset, source_width, source_height - offset)

    source_resized = source_img.crop(resize).resize((target_width, target_height), Image.ANTIALIAS)
    return source_resized


def combine_and_mask(img_new, mask, img_black):
    """
    Combine img_new, mask, and image_black based on the mask

    img_new: new (unmasked image)
    mask: binary mask of bird image
    img_black: already-masked bird image (bird only)
    """
    # Warp new img to match black img
    img_resized = crop_and_resize(img_new, img_black)
    img_resized_np = np.asarray(img_resized)

    # Mask new img
    img_masked_np = np.around(img_resized_np * (1 - mask)).astype(np.uint8)

    # Combine
    img_combined_np = np.asarray(img_black) + img_masked_np
    img_combined = Image.fromarray(img_combined_np)

    return img_combined

def get_places(fname):
    """
    Load list of places imgs and classes into dictionary
    """
    places_dict = defaultdict(list)
    with open(fname, 'r') as f:
        for line in f:
            img_name, n = line.split()
            places_dict[int(n)].append(img_name)
    return places_dict

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Make segmentations',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--cub_dir', default='datasets/CUB_200_2011/', help='Path to CUB (should also contain segmentations folder)')
    parser.add_argument('--places_dir', default='datasets/places365/', help='Path to Places365 dataset')
    parser.add_argument('--places_split', default='val_large', help='Which Places365 split to use (folder in --places_dir)')
    parser.add_argument('--places_file', default='places365_val.txt', help='Filepath to list of places images and classes (file in --places_dir)')
    parser.add_argument('--out_dir', default='.', help='Output directory')
    parser.add_argument('--black_dirname', default='CUB_black', help='Name of black dataset: black background for each image')
    parser.add_argument('--random_dirname', default='CUB_random', help='Name of random dataset: completely random place sampled for each image')
    parser.add_argument('--fixed_dirname', default='CUB_fixed', help='Name of fixed dataset: class <-> place association fixed at train, swapped at test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)

    # Get species
    img_dir = os.path.join(args.cub_dir, 'images')
    seg_dir = os.path.join(args.cub_dir, 'segmentations')
    species = sorted(os.listdir(img_dir))

    # Make output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Get list of places
    places_dict = get_places(os.path.join(args.places_dir, args.places_file))

    # Full paths
    places_dict = {k: [os.path.join(args.places_dir, args.places_split, p) for p in v]
                   for k, v in places_dict.items()}

    # Flat list of places
    all_places = [item for sublist in places_dict.values() for item in sublist]
    assert all(os.path.exists(p) and p.endswith('.jpg') for p in all_places)
    # Iterate through places
    all_places_i = 0
    np.random.shuffle(all_places)

    # Arbitrarily map places class to birds class
    sampled_places = np.random.choice(list(places_dict.keys()), size=len(species), replace=False)
    s2p_train = {s: int(p) for s, p in zip(species, sampled_places)}
    # Shift sampled places at test
    s2p_test = {s: int(p) for s, p in zip(species, np.roll(sampled_places, 1))}

    for spc in species:
        spc_img_dir = os.path.join(img_dir, spc)
        spc_seg_dir = os.path.join(seg_dir, spc)

        # List images in species
        spc_img = sorted(os.listdir(spc_img_dir))
        spc_seg = sorted(os.listdir(spc_seg_dir))

        # Make sure directory files align
        assert all(i.endswith('.jpg') for i in spc_img)
        assert all(i.endswith('.png') for i in spc_seg)
        assert all(os.path.splitext(x)[0] == os.path.splitext(y)[0] for x, y in zip(spc_img, spc_seg))

        # New output directories
        spc_black_dir = os.path.join(args.out_dir, args.black_dirname, spc)
        spc_random_dir = os.path.join(args.out_dir, args.random_dirname, spc)
        spc_train_dir = os.path.join(args.out_dir, args.fixed_dirname, 'train', spc)
        spc_test_dir = os.path.join(args.out_dir, args.fixed_dirname, 'test', spc)

        os.makedirs(spc_black_dir, exist_ok=True)
        os.makedirs(spc_random_dir, exist_ok=True)
        os.makedirs(spc_train_dir, exist_ok=True)
        os.makedirs(spc_test_dir, exist_ok=True)

        # Get fixed places for this species
        train_place = s2p_train[spc]
        test_place = s2p_test[spc]
        train_place_imgs = np.random.choice(places_dict[train_place], size=len(spc_img), replace=False)
        test_place_imgs = np.random.choice(places_dict[test_place], size=len(spc_img), replace=False)

        # (image, segmentation, train place, test place
        it = zip(spc_img, spc_seg, train_place_imgs, test_place_imgs)

        for img_path, seg_path, train_place_path, test_place_path in it:
            full_img_path = os.path.join(spc_img_dir, img_path)
            full_seg_path = os.path.join(spc_seg_dir, seg_path)

            # Load images
            img_np = np.asarray(Image.open(full_img_path).convert('RGB'))
            # Turn into opacity filter
            seg_np = np.asarray(Image.open(full_seg_path).convert('RGB')) / 255

            # Black background
            img_black_np = np.around(img_np * seg_np).astype(np.uint8)

            full_black_path = os.path.join(spc_black_dir, img_path)
            img_black = Image.fromarray(img_black_np)
            img_black.save(full_black_path)

            # Random background
            random_place_path = all_places[all_places_i]
            all_places_i += 1
            random_place = Image.open(random_place_path).convert('RGB')

            img_random = combine_and_mask(random_place, seg_np, img_black)
            full_random_path = os.path.join(spc_random_dir, img_path)
            img_random.save(full_random_path)

            # Fixed background
            train_place = Image.open(train_place_path).convert('RGB')
            test_place = Image.open(test_place_path).convert('RGB')

            img_train = combine_and_mask(train_place, seg_np, img_black)
            img_test = combine_and_mask(test_place, seg_np, img_black)

            full_train_path = os.path.join(spc_train_dir, img_path)
            img_train.save(full_train_path)
            full_test_path = os.path.join(spc_test_dir, img_path)
            img_test.save(full_test_path)

    # Save fixed class/image metadata
    # TODO: Should probably record individual places images too
    fixed_dir = os.path.join(args.out_dir, args.fixed_dirname)
    with open(os.path.join(fixed_dir, 'train_places.json'), 'w') as f:
        json.dump(s2p_train, f, sort_keys=True, indent=4)
    with open(os.path.join(fixed_dir, 'test_places.json'), 'w') as f:
        json.dump(s2p_test, f, sort_keys=True, indent=4)
