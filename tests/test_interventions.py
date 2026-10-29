"""Engineering witnesses, never empirical faithfulness results."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from analysis.intervention_analysis import analyze, CONDITIONS
from scripts.build_interventions import build, matched_control


def planned(source='s1', image='i1', claim='c1', support=('true','false','true')):
    return dict(source_group=source, image_id=image, claim_id=claim, claim_text='The wheel is deformed.',
                validity_status='complete', validity_annotator_id='input_rater',
                **dict(zip(('intact_supported','referent_supported','control_supported'), support)))


def outputs(plan, pattern=(1,0,1), arm='G'):
    return [dict(source_group=plan['source_group'], image_id=plan['image_id'], claim_id=plan['claim_id'],
                 arm=arm, intervention=c, status='complete', claim_asserted=str(a),
                 output_text='The wheel is deformed.' if a else 'Insufficient evidence.', annotator_id='blinded_rater')
            for c,a in zip(CONDITIONS, pattern)]


class AnalysisTests(unittest.TestCase):
    def run_case(self, pattern):
        p=planned()
        return analyze([p],outputs(p,pattern),bootstrap=20)

    def test_perfect_directional_response_is_not_zero(self):
        result=self.run_case((1,0,1))
        self.assertEqual(result['status'],'complete')
        for name in ('joint_evidence_pattern','referent_withdrawal','control_retention','selective_assertion_drop'):
            self.assertEqual(result['estimates'][name]['source_macro_mean'],1)
        self.assertIsNone(result['estimates']['joint_evidence_pattern']['source_cluster_ci95'])

    def test_constant_or_indiscriminate_responses_do_not_pass(self):
        for pattern in ((1,1,1),(0,0,0),(1,0,0),(0,0,1)):
            with self.subTest(pattern=pattern):
                self.assertEqual(self.run_case(pattern)['estimates']['joint_evidence_pattern']['source_macro_mean'],0)
        # Selectivity alone rewards this artifact; the joint metric must reject it.
        self.assertEqual(self.run_case((0,0,1))['estimates']['selective_assertion_drop']['source_macro_mean'],1)

    def test_all_eight_patterns_have_one_joint_success(self):
        scores=[]
        for a in (0,1):
            for b in (0,1):
                for c in (0,1):
                    scores.append(self.run_case((a,b,c))['estimates']['joint_evidence_pattern']['source_macro_mean'])
        self.assertEqual(sum(scores),1)

    def test_entire_missing_claim_cannot_disappear(self):
        a,b=planned(),planned('s2','i2')
        result=analyze([a,b],outputs(a),bootstrap=20)
        self.assertEqual(result['status'],'incomplete_denominator')
        self.assertEqual(result['planned_rows'],6)
        self.assertEqual(len(result['issues']),3)
        self.assertIsNone(result['estimates'])
        self.assertEqual(result['analyzed_claims'],1)

    def test_empty_annotations_and_missing_intact_are_incomplete(self):
        p=planned()
        for rows in ([],outputs(p)[1:]):
            result=analyze([p],rows,bootstrap=20)
            self.assertEqual(result['status'],'incomplete_denominator')
            self.assertEqual(result['analyzed_claims'],0)

    def test_other_arm_failure_does_not_contaminate(self):
        p=planned()
        extra=outputs(p,arm='baseline')
        extra[0]['status']='failed'
        self.assertEqual(analyze([p],outputs(p)+extra,bootstrap=20)['status'],'complete')

    def test_input_validity_controls_eligibility_not_output(self):
        p=planned(support=('true','true','true'))
        result=analyze([p],outputs(p),bootstrap=20)
        self.assertEqual(result['status'],'no_eligible_claims')
        self.assertEqual(len(result['excluded_by_input_validity']),1)
        p['validity_status']='pending'
        self.assertEqual(analyze([p],outputs(p),bootstrap=20)['status'],'incomplete_denominator')

    def test_invalid_annotation_is_not_a_negative_label(self):
        p=planned()
        for value in ('',None,'maybe'):
            rows=outputs(p)
            rows[0]['claim_asserted']=value
            self.assertEqual(analyze([p],rows,bootstrap=20)['status'],'incomplete_denominator')

    def test_null_text_and_annotator_fields_cannot_validate(self):
        p=planned()
        for field in ('claim_text','source_group','image_id','claim_id'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyze([{**p,field:None}],outputs(p),bootstrap=20)
        result=analyze([{**p,'validity_annotator_id':None}],outputs(p),bootstrap=20)
        self.assertEqual(result['status'],'incomplete_denominator')
        for field in ('output_text','annotator_id'):
            rows=outputs(p);rows[0][field]=None
            self.assertEqual(analyze([p],rows,bootstrap=20)['status'],'incomplete_denominator')

    def test_empty_text_cannot_assert_but_can_abstain(self):
        p=planned();rows=outputs(p)
        rows[0]['output_text']='   '
        self.assertEqual(analyze([p],rows,bootstrap=20)['status'],'incomplete_denominator')
        rows=outputs(p);rows[1]['output_text']=''
        self.assertEqual(analyze([p],rows,bootstrap=20)['status'],'complete')

    def test_short_csv_row_cannot_supply_an_annotator(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=planned();rows=outputs(p)
            for name,records in [('plan.csv',[p]),('outputs.csv',rows)]:
                with (root/name).open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=records[0].keys())
                    writer.writeheader();writer.writerows(records)
            # A real CSV short row becomes annotator_id=None in DictReader.
            text=(root/'outputs.csv').read_text().replace(',blinded_rater','',1)
            (root/'outputs.csv').write_text(text)
            out=root/'result.json'
            r=subprocess.run([sys.executable,'analysis/intervention_analysis.py',str(root/'outputs.csv'),
                              '--manifest',str(root/'plan.csv'),'--bootstrap','20','--output',str(out)],
                             capture_output=True,text=True)
            self.assertEqual(r.returncode,2,r.stderr)
            self.assertIsNone(json.loads(out.read_text())['estimates'])

    def test_source_macro_and_claim_macro_are_distinct(self):
        plans=[planned('large','i1'),planned('large','i2'),planned('small','i3')]
        rows=outputs(plans[0])+outputs(plans[1])+outputs(plans[2],(1,1,1))
        r=analyze(plans,rows,bootstrap=100)
        e=r['estimates']['joint_evidence_pattern']
        self.assertEqual(e['source_macro_mean'],.5)
        self.assertAlmostEqual(e['claim_macro_mean'],2/3)
        self.assertEqual(e['source_cluster_ci95'],[0,1])
        self.assertEqual(r,analyze(plans,rows,bootstrap=100))

    def test_invalid_configuration_and_identity_fail(self):
        p=planned()
        for plans,rows,n in (([],[],20),([p],outputs(p),0),([p,p],outputs(p),20),
                             ([p],outputs(p)*2,20),([p],outputs(planned('s2')),20)):
            with self.assertRaises(ValueError):
                analyze(plans,rows,bootstrap=n)

    def test_cli_incomplete_result_has_nonzero_exit_and_no_headline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            p=planned()
            for name,rows in [('plan.csv',[p]),('outputs.csv',outputs(p)[1:])]:
                with (root/name).open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=rows[0].keys())
                    writer.writeheader(); writer.writerows(rows)
            out=root/'result.json'
            r=subprocess.run([sys.executable,'analysis/intervention_analysis.py',str(root/'outputs.csv'),
                              '--manifest',str(root/'plan.csv'),'--bootstrap','20','--output',str(out)],
                             capture_output=True,text=True)
            self.assertEqual(r.returncode,2,r.stderr)
            self.assertIsNone(json.loads(out.read_text())['estimates'])


class GenerationTests(unittest.TestCase):
    def test_same_shape_area_disjoint_including_single_pixel(self):
        for coords in (((1,1),),((1,1),(1,2),(2,1))):
            m=np.zeros((8,8),dtype=bool)
            for y,x in coords: m[y,x]=True
            c,shift=matched_control(m,('s','i','c'),300)
            self.assertEqual(int(c.sum()),len(coords))
            self.assertFalse((c&m).any())
            expected={(y+shift['dy'],x+shift['dx']) for y,x in coords}
            self.assertEqual(set(map(tuple,np.argwhere(c))),expected)
            np.testing.assert_array_equal(c,matched_control(m,('s','i','c'),300)[0])

    def fixture(self,root,mask):
        array=np.arange(8*8*3,dtype=np.uint8).reshape(8,8,3)
        Image.fromarray(array).save(root/'input.png')
        Image.fromarray(mask).save(root/'mask.png')
        rows=[dict(source_group='same_source',image_id=i,claim_id='same_claim',claim_text='A defect.',
                   image_path='input.png',mask_path='mask.png') for i in ('image1','image2')]
        manifest=root/'claims.csv'
        with manifest.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys())
            writer.writeheader();writer.writerows(rows)
        return manifest,array

    def test_end_to_end_intact_preserved_and_images_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); mask=np.zeros((8,8),dtype=np.uint8);mask[1:3,1]=255;mask[1,2]=255
            manifest,array=self.fixture(root,mask)
            result=build(manifest,root/'out')
            self.assertEqual(result['complete_claims'],2)
            rows=[json.loads(l) for l in (root/'out/manifest.jsonl').read_text().splitlines()]
            self.assertEqual(len({r['output_path'] for r in rows}),6)
            for row in rows:
                image=np.array(Image.open(row['output_path']))
                region=np.array(Image.open(row['intervention_mask_path']))>0
                if row['intervention']=='intact':
                    np.testing.assert_array_equal(image,array)
                    self.assertEqual(row['occluded_pixels'],0)
                else:
                    self.assertEqual(row['occluded_pixels'],3)
                    np.testing.assert_array_equal(image[~region],array[~region])
                if row['intervention']=='matched_control_occlusion':
                    self.assertEqual(row['referent_overlap_pixels'],0)
            with (root/'out/validity_template.csv').open() as f:
                labels=list(csv.DictReader(f))
            self.assertTrue(all(r['validity_status']=='pending' and r['intact_supported']=='' for r in labels))
            build(manifest,root/'repeat')
            repeat=[json.loads(l) for l in (root/'repeat/manifest.jsonl').read_text().splitlines()]
            self.assertEqual([r['output_sha256'] for r in rows],[r['output_sha256'] for r in repeat])

    def test_bad_alignment_or_impossible_control_retains_all_rows(self):
        for mask,reason in ((np.ones((4,4),dtype=np.uint8)*255,'mask_image_size_mismatch'),
                            (np.ones((8,8),dtype=np.uint8)*255,'control_search_exhausted'),
                            (np.zeros((8,8),dtype=np.uint8),'empty_referent_mask')):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as d:
                root=Path(d);manifest,_=self.fixture(root,mask)
                result=build(manifest,root/'out')
                self.assertEqual(result['status'],'incomplete_denominator')
                self.assertEqual(result['failed_claims'],2)
                rows=[json.loads(l) for l in (root/'out/manifest.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),6)
                self.assertTrue(all(reason in r['failure'] for r in rows))

    def test_empty_or_duplicate_manifest_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest,_=self.fixture(root,np.zeros((8,8),dtype=np.uint8))
            lines=manifest.read_text().splitlines()
            for content in ('\n'.join(lines[:1])+'\n','\n'.join([lines[0],lines[1],lines[1]])+'\n'):
                manifest.write_text(content)
                with self.assertRaises(ValueError):build(manifest,root/'out')
                self.assertFalse((root/'out').exists())


if __name__=='__main__':
    unittest.main()
