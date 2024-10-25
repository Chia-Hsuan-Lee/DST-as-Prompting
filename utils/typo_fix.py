# from IC-DST 
# https://github.com/Yushi-Hu/IC-DST/blob/main/utils/typo_fix.py

def check_prefix_suffix(value, candidates):
    # add/delete "the" in the front, or the suffix in the end.
    if value in candidates:
        return value
    prefixes = ['the ']
    suffixes = [" hotel", " restaurant", ' cinema', ' guest house',
                " theatre", " airport", " street", ' gallery', ' museum']
    for prefix in prefixes:
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    for suffix in suffixes:
        if value.endswith(suffix):
            value = value[:-len(suffix)]
            break
    for prefix in [''] + prefixes:
        for suffix in [''] + suffixes:
            possible_value = prefix + value + suffix
            if possible_value in candidates:
                return possible_value
    return ''

def typo_fix(slot_values, ontology, version="2.1"):

    # fix the named entities in these slots
    named_entity_slots = ['hotel-name', 'train-destination', 'train-departure',
                          'attraction-type', 'attraction-name',
                          'restaurant-name', 'taxi-departure', 'taxi-destination', 'restaurant-food']
    fixed = {}
    for slot, value in slot_values.items():
        # fix 's
        value = value.replace(' s ', 's ')
        if value.endswith(' s'):
            value = value[:-2] + 's'

        # fix typo words
        general_typos = {'fen ditton': 'fenditton',
                         'guesthouse': 'guest house',
                         'steveage': 'stevenage',
                         'stantsted': 'stansted',
                         'storthford': 'stortford',
                         'shortford': 'stortford',
                         'weish': 'welsh',
                         'bringham': 'birmingham',
                         'liverpoool': 'liverpool',
                         'petersborough': 'peterborough',
                         'el shaddai': 'el shaddia',
                         'wendesday': 'wednesday',
                         'brazliian': 'brazilian',
                         'graffton': 'grafton'}
        for k, v in general_typos.items():
            value = value.replace(k, v)

        # fix whole value
        value_replacement = {'center': 'centre',
                             'caffe uno': 'cafe uno',
                             'caffee uno': 'cafe uno',
                             'christs college': 'christ college',
                             'churchill college': 'churchills college',
                             'sat': 'saturday',
                             'saint johns chop shop house': 'saint johns chop house',
                             'good luck chinese food takeaway': 'good luck',
                             'asian': 'asian oriental',
                             'gallery at 12': 'gallery at 12 a high street'}

        if version == "2.1":
            value_replacement['portuguese'] = 'portugese'
            value_replacement['museum of archaeology and anthropology'] = 'museum of archaelogy and anthropology'

        if version == "2.4":
            value_replacement['portugese'] = 'portuguese'
            value_replacement['museum of archaelogy and anthropology'] = 'museum of archaeology and anthropology'

        for k, v in value_replacement.items():
            if value == k:
                value = v

        # time format fix  9:00 -> 09:00
        if ':' in value and len(value) < 5:
            value = '0' + value

        if slot in named_entity_slots:
            value = check_prefix_suffix(value, ontology[slot])

        if value:
            fixed[slot] = value
    return fixed