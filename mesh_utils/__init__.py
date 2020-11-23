# Define names for each of the tags in the model. s
Tags = dict(
    background=0,
    lv_myocardium=1,
    rv_myocardium=2,
    la_myocardium=3,
    ra_myocardium=4,
    aorta=5,
    pulmonary_artery=6,
    mitral_valve=7,
    tricuspid_valve=8,
    aortic_valve=9,
    pulmonary_valve=10,
    appendage=11,
    left_superior_pulmonary_vein=12,
    left_inferiror_pulmonary_vein=13,
    right_inferior_pulmonary_vein=14,
    right_superior_pulmonary_vein=16,
    superior_vena_cava=16,
    inferior_vena_cava=17,
    appendage_border=18,
    right_inferior_pulmonary_vein_border=19,
    left_inferior_pulmonary_vein_border=20,
    left_superior_pulmonary_vein_border=21,
    right_superior_pulmonary_vein_border=22,
    superior_vena_cava_border=23,
    inferior_vena_cava_border=24,
    pericardium=25,
    # Below here is not included in the mesh by default but defined here to be added to the mesh
    lv_blood_pool=31,
    rv_blood_pool=32,
    la_blood_pool=33,
    ra_blood_pool=34,
    aorta_blood_pool=35,
    label_background=36,  # do differentiate ultrasound cone in the label image
    inside_background=37,
    # special label for things inside the heart but still background (to keepy outside as 0 which is fill val)
)

# 11-17 are "valves" at the end of inputs/outputs
# 18-24 are "caps" at the end of inputs/outputs - include in tissue

TissueTagNames = [t for t in Tags if any([name in t for name in ['myocardium', 'border']])]
ValveTagNames = [t for t in Tags if t not in TissueTagNames and "valve" in t]
OtherTagNames = [t for t in Tags if
                 t not in TissueTagNames and any([name in t for name in ['appendage', 'vein', 'cava']])]
BloodTagNames = [t for t in Tags if any([name in t for name in ['pool']])]
