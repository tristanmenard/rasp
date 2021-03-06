"""Make TAFL data codes human-readable"""

def service_code(code):
    if code == 1:
        return 'Broadcast'
    elif code == 2:
        return 'Fixed'
    elif code == 3:
        return 'Land Mobile'
    elif code == 5:
        return 'Maritime'
    elif code == 8:
        return 'Aeronautical'
    elif code == 9:
        return 'Satellite'
    elif code == 85:
        return 'Spectrum License'
    else:
        return 'Unknown'

def subservice_code(code):
    if code == 19:
        return 'AM'
    elif code == 20:
        return 'Shortwave Radio'
    elif code == 21:
        return 'Low Power FM'
    elif code == 25:
        return 'FM'
    elif code == 26:
        return 'Analog TV'
    elif code == 27:
        return 'Lower Power DTV'
    elif code == 28:
        return 'Digital Satellite Radio'
    elif code == 29:
        return 'DTV'
    elif code == 30:
        return 'Multipoint Distribution TV'
    elif code == 200:
        return 'Point-to-Point'
    elif code == 201:
        return 'Point-to-Multipoint'
    elif code == 202:
        return 'Point-to-Transportable'
    elif code == 203:
        return 'Transportable-to-Transportable'
    elif code == 300:
        return 'Base-Mobile Systems'
    elif code == 301:
        return 'Light-licensed Mobile'
    elif code == 302:
        return 'Mobile Only'
    elif code == 304:
        return 'Radiodetermination'
    elif code == 500:
        return 'Maritime Base Station'
    elif code == 501:
        return 'Marine Vessel'
    elif code == 640:
        return 'Passive Reflector'
    elif code == 641:
        return 'Passive Repeater'
    elif code == 800:
        return 'Aeronautical Base Stations'
    elif code == 801:
        return 'Aircraft'
    elif code == 870:
        return 'Auction'
    elif code == 871:
        return 'Non-Auction'
    elif code == 872:
        return 'Subordinate'
    elif code == 873:
        return 'RP-019'
    elif code in (900,901,902,903,904):
        return 'Earth Station'
    elif code == 905:
        return 'Satellite'
    else:
        return 'Unknown'
