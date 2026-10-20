"""Exact official TRAIN join and clean explanation serialization; no generated labels."""
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np
import torch
from pycocotools import mask as coco_mask

IMAGE_TOKEN=-200
SPECIAL_STRINGS=('<image>','<im_start>','<im_end>','<p>','</p>','[SEG]','</s>','USER:','ASSISTANT:')

class Ineligible(ValueError): pass

def digest_bytes(value): return hashlib.sha256(value).hexdigest()

def load_author_helpers(source_dir):
    root=Path(source_dir)
    conv_path=root/'model/llava/conversation.py'
    module_name='official_clean_explanation_author_conversation_'+digest_bytes(str(root.resolve()).encode())[:12]
    spec=importlib.util.spec_from_file_location(module_name,conv_path)
    module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module;spec.loader.exec_module(module)
    source=(root/'model/llava/mm_utils.py').read_text()
    tree=ast.parse(source);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='tokenizer_image_token')
    namespace={'torch':torch,'IMAGE_TOKEN_INDEX':IMAGE_TOKEN}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(root/'model/llava/mm_utils.py'),'exec'),namespace)
    qt=ast.parse((root/'dataset/utils/utils.py').read_text())
    questions=next(ast.literal_eval(n.value) for n in qt.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GCG_QUESTIONS' for t in n.targets))
    return module.conv_templates['llava_v1'].copy(),namespace['tokenizer_image_token'],tuple(s.strip() for s in questions)


def render(conv_template, question, answer, *, query=None):
    conv=conv_template.copy();conv.messages=[]
    user='The <image> provides an overview of the picture.\n'+question
    if query is not None:
        # Only an official referent, not the image caption or a candidate answer.
        user+='\nFor this response, focus only on the following annotated referent: '+query+'\nExplain the visible artifact in this referent in complete sentences. Return the explanation text only.'
    conv.append_message(conv.roles[0],user)
    conv.append_message(conv.roles[1],answer)
    return conv.get_prompt().replace('<image>','<im_start><image><im_end>')


@dataclass
class SerializedExplanation:
    input_ids:torch.Tensor
    target_begin:int
    target_end:int
    image_index:int
    prefix_text:str
    target_text:str
    full_text:str
    expanded_length:int


def serialize_explanation(*, phrase, explanation, question, tokenizer, conv_template,
                          token_image, allowed_questions, max_expanded_length=1536):
    if question not in allowed_questions: raise Ineligible('question_not_original_author_template')
    if not isinstance(phrase,str) or not phrase.strip(): raise Ineligible('empty_official_referent')
    if not isinstance(explanation,str) or not explanation.strip(): raise Ineligible('empty_official_explanation')
    if any(s in phrase or s in explanation for s in SPECIAL_STRINGS): raise Ineligible('special_control_string_in_official_text')
    prefix=render(conv_template,question,None,query=phrase)
    full=render(conv_template,question,explanation,query=phrase)
    # Author TWO template appends EOS to a nonempty assistant answer. Exclude it
    # from the likelihood target; every official explanation token is retained.
    if not full.endswith(conv_template.sep2):raise ValueError('Unexpected author answer delimiter')
    full=full[:-len(conv_template.sep2)]
    if not full.startswith(prefix+' '):raise ValueError('Author assistant boundary changed')
    prefix_ids=token_image(prefix,tokenizer,return_tensors='pt')
    ids=token_image(full,tokenizer,return_tensors='pt')
    begin=prefix_ids.numel();end=ids.numel()
    if not torch.equal(ids[:begin],prefix_ids):raise Ineligible('unstable_tokenization_at_assistant_boundary')
    if end<=begin:raise Ineligible('empty_tokenized_explanation')
    if int((ids==IMAGE_TOKEN).sum())!=1:raise Ineligible('expected_one_image_placeholder')
    if bool((ids==tokenizer.pad_token_id).any()):raise Ineligible('nontrivial_attention_in_clean_replay')
    image_index=int(torch.nonzero(ids==IMAGE_TOKEN)[0])
    if image_index>=begin:raise ValueError('Image must precede explanation')
    target=ids[begin:end].tolist();pre=ids[:begin].tolist()
    if any(pre[i:i+len(target)]==target for i in range(len(pre)-len(target)+1)):
        raise Ineligible('complete_target_token_sequence_in_clean_prefix')
    if explanation.strip().casefold() in prefix.casefold():raise Ineligible('complete_explanation_text_in_clean_prefix')
    if end+575>max_expanded_length:raise Ineligible('full_explanation_exceeds_context_no_truncation')
    return SerializedExplanation(ids[None],begin,end,image_index,prefix,explanation,full,end+575)


class OfficialTrainIndex:
    def __init__(self, path, expected_sha256, *, strict=True):
        self.path=Path(path);raw=self.path.read_bytes();self.sha256=digest_bytes(raw)
        if self.sha256!=expected_sha256:raise ValueError('Official TRAIN annotation SHA mismatch')
        if strict and not str(self.path).endswith('/SynthScars/train/annotations/train.json'):
            raise ValueError('Only bound official TRAIN annotations are accepted')
        self.by_image={};self.mask_cache={};self.records=0
        seen_ids=set()
        for row in json.loads(raw):
            if not isinstance(row,dict) or len(row)!=1:raise ValueError('Expected single-ID official annotation records')
            aid,item=next(iter(row.items()))
            if aid in seen_ids:raise ValueError('Duplicate annotation ID')
            seen_ids.add(aid)
            caption=item['caption'].strip('"').strip();refs=[]
            for original_index,ref in enumerate(item['refs']):
                phrase=ref['sentence']
                if phrase in caption:
                    begin=caption.find(phrase)
                    refs.append({'original_ref_index':original_index,'phrase':phrase,'explanation':ref.get('explanation'),
                        'segmentation':ref['segmentation'],'begin':begin,'end':begin+len(phrase)})
            refs.sort(key=lambda x:x['begin']) # Same stable start sort as author.
            answer=caption
            for ref in sorted(refs,key=lambda x:x['begin'],reverse=True):
                b,e=ref['begin'],ref['end'];answer=answer[:b]+'<p> '+answer[b:e]+' </p> [SEG]'+answer[e:]
            record={'annotation_id':aid,'caption':caption,'tagged_answer':answer,'refs':refs,'image_name':item['img_file_name']}
            self.by_image.setdefault(item['img_file_name'],[]).append(record);self.records+=1

    def mask_sha(self,record,height,width):
        key=(record['annotation_id'],height,width)
        if key not in self.mask_cache:
            masks=[]
            for ref in record['refs']:
                binary=np.zeros((height,width),dtype=np.uint8)
                for segment in ref['segmentation']:
                    rles=coco_mask.frPyObjects([np.array(segment)],height,width)
                    binary+=coco_mask.decode(rles).astype(np.uint8).squeeze()
                masks.append(binary)
            self.mask_cache[key]=digest_bytes(np.stack(masks).astype(np.float32).tobytes()) if masks else None
        return self.mask_cache[key]

    def join(self,batch,*,tokenizer,conv_template,token_image,allowed_questions,max_expanded_length,strict=True):
        if batch['input_ids'].shape[0]!=1 or len(batch['image_paths'])!=1 or len(batch['conversation_list'])!=1:
            raise Ineligible('expected_one_image_one_original_conversation')
        if batch['offset'].detach().cpu().tolist()!=[0,1] or batch.get('bboxes') is not None:
            raise Ineligible('unexpected_offset_or_region_bypass')
        path=str(batch['image_paths'][0])
        if strict and '/SynthScars/train/images/' not in path:raise Ineligible('nonofficial_train_source')
        questions=batch.get('questions_list')
        if not questions or len(questions)!=1 or len(questions[0])!=1:raise Ineligible('missing_exact_original_question')
        question=questions[0][0].strip()
        if question not in allowed_questions:raise Ineligible('question_not_original_author_template')
        conversation=batch['conversation_list'][0]
        if not isinstance(conversation,str):raise Ineligible('missing_original_conversation_for_provenance_join')
        if not batch.get('masks_list') or len(batch['masks_list'])!=1:raise Ineligible('missing_official_masks')
        gt=batch['masks_list'][0].detach().float().cpu()
        if gt.ndim!=3 or not len(gt) or not bool(torch.isfinite(gt).all()) or not bool((gt>=0).all()):
            raise Ineligible('invalid_official_masks')
        mask_sha=digest_bytes(gt.numpy().tobytes());matches=[]
        for record in self.by_image.get(Path(path).name,[]):
            if render(conv_template,question,record['tagged_answer'])!=conversation:continue
            if len(record['refs'])!=len(gt):continue
            if self.mask_sha(record,gt.shape[1],gt.shape[2])!=mask_sha:continue
            matches.append(record)
        if len(matches)!=1:raise Ineligible('ambiguous_or_missing_exact_annotation_caption_mask_join')
        expected=token_image(conversation,tokenizer,return_tensors='pt')[:max_expanded_length-575]
        actual=batch['input_ids'][0].detach().cpu()
        if not torch.equal(expected,actual):raise Ineligible('native_tokenized_conversation_mismatch')
        attention=batch.get('attention_masks')
        if not isinstance(attention,torch.Tensor) or attention.shape!=batch['input_ids'].shape:
            raise Ineligible('missing_native_attention_contract')
        if not torch.equal(attention[0].detach().bool().cpu(),actual.ne(tokenizer.pad_token_id)):
            raise Ineligible('native_attention_contract_mismatch')
        return matches[0],question,gt,{'annotation_id':matches[0]['annotation_id'],
            'native_conversation_sha256':digest_bytes(conversation.encode()),'native_masks_sha256':mask_sha,
            'official_annotation_sha256':self.sha256,'native_exact_join':True}


def native_supervision_binding(record,ref,mask_index,batch,*,question,tokenizer,conv_template,token_image):
    """Require this complete explanation AND its SEG in the native retained labels.

The current native caption is used only to prove same-batch supervision coverage;
none of its tokens become the clean auxiliary prefix. Boundary ambiguity skips.
"""
    ids=batch['input_ids'][0].detach().cpu();labels=batch['labels'][0].detach().cpu()
    if labels.shape!=ids.shape:raise Ineligible('native_labels_shape_mismatch')
    seg_positions=torch.nonzero(ids==32004).flatten()
    if mask_index>=len(seg_positions):raise Ineligible('selected_ref_SEG_truncated_from_native_batch')
    seg_index=int(seg_positions[mask_index])
    if int(labels[seg_index])!=32004:raise Ineligible('selected_ref_SEG_not_native_supervised')
    explanation=ref.get('explanation')
    if not isinstance(explanation,str) or not explanation.strip():raise Ineligible('empty_official_explanation')
    answer=record['tagged_answer']
    marker='<p> '+ref['phrase']+' </p> [SEG]'
    if answer.count(marker)!=1:raise Ineligible('ambiguous_native_query_marker')
    start=answer.find(explanation,answer.index(marker)+len(marker))
    if start<0:raise Ineligible('official_explanation_not_intact_after_referent_in_native_caption')
    if answer.find(explanation,start+1)>=0:raise Ineligible('ambiguous_explanation_occurrence_after_referent')
    end=start+len(explanation)
    def token_boundary(char_end):
        text=render(conv_template,question,answer[:char_end])
        if not text.endswith(conv_template.sep2):raise ValueError('Unexpected native rendered EOS')
        return token_image(text[:-len(conv_template.sep2)],tokenizer,return_tensors='pt')
    prefix_start=token_boundary(start);prefix_end=token_boundary(end)
    b,e=len(prefix_start),len(prefix_end)
    if e>len(ids):raise Ineligible('full_explanation_truncated_from_native_supervision')
    if not torch.equal(ids[:b],prefix_start) or not torch.equal(ids[:e],prefix_end):
        raise Ineligible('native_explanation_token_boundary_ambiguous')
    if b>=e or not torch.equal(labels[b:e],ids[b:e]):raise Ineligible('full_explanation_not_native_supervised')
    return {'native_ref_SEG_position':seg_index,'native_explanation_token_begin':b,
        'native_explanation_token_end':e,'native_explanation_fully_supervised':True}
