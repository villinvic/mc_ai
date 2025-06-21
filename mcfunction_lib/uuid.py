def convert_nbt_uuid_to_hex(nbt_uuid):
    return '-'.join(f"{int(part) & 0xFFFFFFFF:x}" for part in nbt_uuid[3:-1].split(","))

def convert_hex_to_nbt_uuid(hex_str):
    parts = hex_str.split('-')
    nbt = str([int(part, 16) for part in parts])
    nbt = nbt[:1] + "I;" + nbt[1:]
    return nbt.replace(" ", "")

if __name__ == '__main__':

    name = convert_nbt_uuid_to_hex(uuid)

    uuid_r = convert_hex_to_nbt_uuid(name)

    print(name, uuid_r)

