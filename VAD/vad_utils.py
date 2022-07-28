# -*- coding: utf-8 -*-


def get_vadtime(varray, sil2voice=15, voice2sil=30, extend_left=10, extend_right=10, max_voicelen=512,
                        min_voicelen=80):
    assert (0 <= extend_left < sil2voice and 0 <= extend_right < voice2sil and max(sil2voice + 20,
                                                                                   sil2voice + voice2sil) <= min_voicelen < max_voicelen)
    st = 0
    ed = 0
    between_sil = 0  # for sil len in two voice sig
    outlist = []
    i = 0

    while i < len(varray) - 20:  # frame range is len(varray)
        vn = 0
        sn = 0

        while i < len(varray) - 20 and vn < sil2voice:
            if varray[i] == 1:
                vn += 1
            else:
                sn += 1
            if sn > voice2sil and sn > vn:
                while i < len(varray) and varray[i] == 0:
                    i += 1

                sn = 0
                vn = 0
                st = i
            i += 1

        if i >= len(varray) - 20:
            break
        st = max(ed - min(between_sil, extend_left),
                    i - vn - 1 - extend_left)
        if len(varray) - st <= min_voicelen:
            outlist += [[st, len(varray)]]
            break

        ed_list = []
        voice_n = 0
        j = st + min_voicelen
        while j < len(varray) and j < st + max_voicelen:
            if varray[j] == 0:
                nt = 0
                while j < len(varray) and varray[j] == 0:
                    nt += 1
                    j += 1
                ed_list += [[j, nt]]
            else:
                voice_n += 1
                j += 1
        if len(ed_list) == 0:  # all is voice
            if len(varray) - (st + max_voicelen) < voice2sil + sil2voice:
                ed = len(varray)
            else:
                ed = min(st + (min_voicelen + max_voicelen) //
                            2, len(varray))
            between_sil = 0
            i = ed
        elif voice_n < voice2sil + sil2voice:  # all is sil
            if ed_list[0][1] > voice2sil:
                ed = ed_list[0][0] - ed_list[0][1] + extend_right
            else:
                ed = ed_list[0][0] - ed_list[0][1] + voice_n
            i = max(ed, ed_list[-1][0])
            between_sil = ed_list[-1][1] // 2
        elif ed_list[0][1] > voice2sil:  # index 0 is long sil
            ed = ed_list[0][0] - ed_list[0][1] + extend_right
            i = ed_list[0][0]
            between_sil = ed_list[0][1] // 2
        else:  # sel the biggest sil
            ed_list.sort(key=lambda d: d[1])
            ed = ed_list[-1][0] - ed_list[-1][1] + \
                min(ed_list[-1][1], extend_right)
            i = ed_list[-1][0]
            between_sil = ed_list[-1][1] // 2
        if ed > len(varray):
            ed = len(varray)
        outlist += [[st, ed]]
        st = 0
        nt = 0
    return outlist
