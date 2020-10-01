docker run -it --rm -v $PWD:/tmp -w /tmp skywills87/card-detect-base python ./id_card_detection_image.py


docker run -it --rm \
-v $PWD:/tmp \
-v $PWD:/tmp/source \
-v $PWD:/tmp/target/ \
-w /tmp \
skywills87/card-detect-base python ./cropped.py -s /tmp/source -t /tmp/target


docker run -it --rm \
-v $PWD:/tmp \
-v /Users/williamkhoo/Desktop/projects/main/mxw/kyc/edge_detect/cn:/tmp/source \
-v /Users/williamkhoo/Desktop/projects/main/mxw/kyc/cn_cropped2:/tmp/target/ \
-w /tmp \
skywills87/card-detect-base python ./cropped.py -s /tmp/source -t /tmp/target