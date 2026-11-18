from math import pi,isclose
from itertools import combinations

savfile=input('.sav file: ')
rasfile=input('.ras file: ')

def family(key):
    return key.split("-")[0]

ALLOWED = (
    "*CORR_POS_VALUE",
    "*HW_GONIOMETER_RADIUS",
    "*MEAS_COND_AXIS_POSITION",
    "*HW_XG_FOCUS",
)

def parse_ras_header(filename):

    h={}

    with open(
        filename,
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in map(str.strip,f):

            if line=="*RAS_HEADER_END":
                break

            if line.startswith("*") and " " in line:

                k,v=line.split(" ",1)

                h[k]=v.strip('" ')

    return h


def parse_sav(filename):

    d={}

    with open(filename) as f:

        for line in map(str.strip,f):

            if (
                not line
                or "=" not in line
                or line.startswith("%")
            ):
                continue

            k,v=line.split("=",1)

            if k.startswith("zweiTheta["):
                continue

            d[k]=v

    return d


def header_numbers(h):

    out=[]

    for k,v in h.items():

        if not k.startswith(ALLOWED):
            continue

        # float

        try:
            out.append(
                (k,float(v))
            )
            continue

        except:
            pass

        # mm

        try:

            if v.endswith("mm"):

                out.append(
                    (k,float(v[:-2]))
                )

                continue

        except:
            pass


        # focus

        try:

            if " x " in v and k == "*HW_XG_FOCUS":

                w,h = v.replace("mm","").split(" x ")

                out.append(
                    (k+" [FocusH]", float(h))
                )

                out.append(
                    (k+" [FocusW]", float(w)/10)
                )

        except:
            pass

        # deg

        try:

            if v.endswith("deg"):

                deg=float(
                    v.replace("deg","")
                     .split("_")[-1]
                )

                out.append(
                    (k,deg)
                )

                out.append(
                    (
                        k+" [rad]",
                        deg*pi/180
                    )
                )

        except:
            pass

    return out


def find_match(value,numbers):

    try:
        target=float(value)
    except:
        return None

    #
    # direct
    #

    for k,v in numbers:

        if isclose(
            target,
            v,
            rel_tol=1e-4,
            abs_tol=1e-4,
        ):
            return (
                "DIRECT",
                k,
                v
            )

    #
    # addition / subtraction
    #

    for (k1,v1),(k2,v2) in combinations(numbers,2):

        if family(k1) != family(k2):
            continue

        tests = (
            ("SUB", f"{k1} - {k2}", v1-v2),
            ("SUB", f"{k2} - {k1}", v2-v1),
        )

        for kind,expr,result in tests:

            if isclose(
                target,
                result,
                rel_tol=1e-4,
                abs_tol=1e-4,
            ):
                return (
                    kind,
                    expr,
                    result,
                )

    return None


def check(savfile,rasfile):

    sav=parse_sav(savfile)

    hdr=parse_ras_header(rasfile)

    nums=header_numbers(hdr)

    print()
    print(savfile)
    print("-"*70)

    for k,v in sav.items():

        m=find_match(v,nums)

        if m:

            kind,src,val=m

            print(
                f"{k:12s}"
                f"{v:12s}"
                f"{kind:8s}"
                f"{src}"
            )

        else:

            print(
                f"{k:12s}"
                f"{v:12s}"
                f"NONE"
            )


# example

check(
    savfile,
    rasfile
)
