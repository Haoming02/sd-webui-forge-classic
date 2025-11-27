import os.path

folder = os.path.dirname(__file__)  # huggingface


class Token:
    wan_compress = os.path.join(folder, "wan.tokenizer.json.xz")
    wan_t2v = os.path.join(folder, "Wan-AI", "Wan2.1-T2V-14B", "tokenizer", "tokenizer.json")
    wan_i2v = os.path.join(folder, "Wan-AI", "Wan2.1-I2V-14B", "tokenizer", "tokenizer.json")

    neta_compress = os.path.join(folder, "neta.tokenizer.json.xz")
    neta_lumina = os.path.join(folder, "neta-art", "Neta-Lumina", "tokenizer", "tokenizer.json")

    z_compress = os.path.join(folder, "z.tokenizer.json.xz")
    z_image = os.path.join(folder, "Tongyi-MAI", "Z-Image-Turbo", "tokenizer", "tokenizer.json")


class sha256:
    wan = "20a46ac256746594ed7e1e3ef733b83fbc5a6f0922aa7480eda961743de080ef"
    # https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/blob/main/tokenizer/tokenizer.json

    neta = "3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922"
    # https://huggingface.co/neta-art/Neta-Lumina-diffusers/blob/main/tokenizer/tokenizer.json

    z = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    # https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/tokenizer/tokenizer.json


def decompress(source: str, target: str):
    import lzma

    with lzma.open(source, "rb") as input_file:
        with open(target, "wb") as output_file:
            output_file.write(input_file.read())


def compress(source: str, target: str):
    import lzma

    with open(source, "rb") as input_file:
        with lzma.open(target, "wb") as output_file:
            output_file.write(input_file.read())


def process():
    if not os.path.isfile(Token.wan_t2v):
        decompress(Token.wan_compress, Token.wan_t2v)
        compare_sha256(Token.wan_t2v, sha256.wan)
    if not os.path.isfile(Token.wan_i2v):
        decompress(Token.wan_compress, Token.wan_i2v)
        compare_sha256(Token.wan_i2v, sha256.wan)
    if not os.path.isfile(Token.neta_lumina):
        decompress(Token.neta_compress, Token.neta_lumina)
        compare_sha256(Token.neta_lumina, sha256.neta)
    if not os.path.isfile(Token.z_image):
        decompress(Token.z_compress, Token.z_image)
        compare_sha256(Token.z_image, sha256.z)

    # if not os.path.isfile(Token.wan_compress):
    #     compress(Token.wan_t2v, Token.wan_compress)
    # if not os.path.isfile(Token.neta_compress):
    #     compare_sha256(Token.neta_lumina, sha256.neta)
    #     compress(Token.neta_lumina, Token.neta_compress)
    # if not os.path.isfile(Token.z_compress):
    #     compress(Token.z_image, Token.z_compress)


def compare_sha256(path: str, target: str):
    import hashlib

    with open(path, "rb") as f:
        data = f.read()
    _hash = hashlib.sha256(data)
    assert _hash.hexdigest() == target
