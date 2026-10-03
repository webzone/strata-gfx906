import copy
import unittest
from tools.gfx906_prefill_compare import compare_arms


class ComparisonTests(unittest.TestCase):
    def case(self,ids=None):
        return dict(kind='code',target=65536,input_ids_sha256='a'*64,ids=[1,2] if ids is None else ids)

    def test_missing_controls_are_not_equality(self):
        result=compare_arms({'IQ2_XS-r0-c4096-m1-s24-jauto':[self.case()]})[0]
        self.assertIsNone(result['ids_equal_canonical'])
        self.assertIsNone(result['ids_equal_same_chunk'])
        self.assertEqual(result['same_chunk_control_repetitions'],0)

    def test_chunk_drift_is_separated_from_mtp_drift(self):
        refs={'IQ2_XS-r0-c2048-m0-s24-j0':[self.case()],
              'IQ2_XS-r0-c4096-m0-s24-j0':[self.case([1,3])],
              'IQ2_XS-r0-c4096-m1-s24-jauto':[self.case([1,3])]}
        result=compare_arms(refs)[-1]
        self.assertFalse(result['ids_equal_canonical'])
        self.assertTrue(result['ids_equal_same_chunk'])

    def test_all_repetitions_are_required_to_match(self):
        refs={'IQ3_S-r0-c2048-m0-s24-j0':[self.case()],
              'IQ3_S-r1-c2048-m0-s24-j0':[self.case([1,4])],
              'IQ3_S-r2-c2048-m1-s24-j0':[self.case()]}
        result=compare_arms(refs)[-1]
        self.assertEqual(result['canonical_control_repetitions'],2)
        self.assertFalse(result['ids_equal_canonical'])

    def test_changed_missing_or_empty_cases_rejected(self):
        refs={'IQ2_XS-r0-c2048-m0-s24-j0':[self.case()],
              'IQ2_XS-r0-c4096-m1-s24-j0':[self.case()]}
        for mutation in ['hash','count','ids']:
            altered=copy.deepcopy(refs)
            if mutation=='hash':altered['IQ2_XS-r0-c4096-m1-s24-j0'][0]['input_ids_sha256']='b'*64
            if mutation=='count':altered['IQ2_XS-r0-c4096-m1-s24-j0'].append(self.case())
            if mutation=='ids':altered['IQ2_XS-r0-c4096-m1-s24-j0'][0]['ids']=[]
            with self.assertRaises(ValueError):compare_arms(altered)
