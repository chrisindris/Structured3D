#!/bin/bash

TARGET_DIR="${1:-./data/}"
URL_PREFIX=$(cat ./secrets/url_prefix.txt)

# the user can choose which ones to implement
PANORAMA="${PANORAMA:-0}"
PERSPECTIVE_FULL="${PERSPECTIVE_FULL:-0}"
PERSPECTIVE_EMPTY="${PERSPECTIVE_EMPTY:-0}"
BBOX_INSTANCE_ANNOTATIONS="${BBOX_INSTANCE_ANNOTATIONS:-0}"
STRUCTURE_ANNOTATIONS="${STRUCTURE_ANNOTATIONS:-0}"
# the user can specify overall min/max
MIN="${MIN:-0}"
MAX="${MAX:-17}"

download_file() {
    local filename="$1"

    if [ -f "$filename" ]; then
        echo "File $filename already exists, skipping download."
    else
        echo "Downloading $filename to $TARGET_DIR"
        wget -P "$TARGET_DIR" "$filename"
    fi
}

get_data() {
    local prefix="$1"
    local min="$2"
    local max="$3"
    for i in $(seq "$min" "$max"); do
        local filename
        filename="${URL_PREFIX}${prefix}$(printf "%02d" "$i").zip"

        download_file "$filename"

    done
}

get_panorama() {
    local min="${1:-0}"
    local max="${2:-17}"
    get_data "Structured3D_panorama_" "$min" "$max"
}

get_perspective_full() {
    local min="${1:-0}"
    local max="${2:-17}"

    # note that file 9 is unavailable, as per https://github.com/bertjiazheng/Structured3D/issues/30
    get_data "Structured3D_perspective_full_" "$min" $(( 8*(max>=8) + max*(max<8) ))
    get_data "Structured3D_perspective_full_" $(( 10*(min<=10) + min*(min>10) )) "$max"
}

get_perspective_empty() {
    local min="${1:-0}"
    local max="${2:-17}"
    get_data "Structured3D_perspective_empty_" "$min" "$max"
}

get_3DBBox_and_instance_annotations() {
    local filename
    filename="${URL_PREFIX}Structured3D_bbox.zip"

    download_file "$filename"

}

get_structure_annotations() {
    local filename
    filename="${URL_PREFIX}Structured3D_annotation_3d.zip"

    download_file "$filename"

}



if [ "$PANORAMA" -eq 1 ]; then
    get_panorama "$MIN" "$MAX"
fi

if [ "$PERSPECTIVE_FULL" -eq 1 ]; then
    get_perspective_full "$MIN" "$MAX"
fi

if [ "$PERSPECTIVE_EMPTY" -eq 1 ]; then
    get_perspective_empty "$MIN" "$MAX"
fi

if [ "$BBOX_INSTANCE_ANNOTATIONS" -eq 1 ]; then
    get_3DBBox_and_instance_annotations
fi

if [ "$STRUCTURE_ANNOTATIONS" -eq 1 ]; then
    get_structure_annotations
fi