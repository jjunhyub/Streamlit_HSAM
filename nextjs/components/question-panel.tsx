// @ts-nocheck
'use client';

import {
  getAnswersBucket,
  getInspectorPills,
  nodeQuestionsFor,
  translatedLabel,
  treeQuestionsFor,
} from '../lib/review-logic';

function SingleChoiceQuestion({ question, value, onChange }) {
  return (
    <div className="questionCard">
      <div className="questionLabel">{question.label}</div>
      <div className="choiceRow">
        {question.options.map((option) => (
          <label className="choiceChip" key={option}>
            <input
              type="radio"
              name={question.id}
              checked={value === option}
              onChange={() => onChange(option)}
            />
            <span>{option}</span>
          </label>
        ))}
      </div>
    </div>
  );
}

function MultiChoiceQuestion({ question, value, onChange }) {
  const selected = Array.isArray(value) ? value : [];

  function toggle(option) {
    const next = selected.includes(option)
      ? selected.filter((item) => item !== option)
      : [...selected, option];
    onChange(next);
  }

  return (
    <div className="questionCard">
      <div className="questionLabel">{question.label}</div>
      <div className="choiceRow">
        {question.options.map((option) => (
          <label className="choiceChip" key={option}>
            <input
              type="checkbox"
              checked={selected.includes(option)}
              onChange={() => toggle(option)}
            />
            <span>{option}</span>
          </label>
        ))}
      </div>
    </div>
  );
}

function TextQuestion({ question, value, onChange }) {
  return (
    <div className="questionCard isWide">
      <div className="questionLabel">{question.label}</div>
      <textarea
        className="questionTextarea"
        value={value || ''}
        onChange={(event) => onChange(event.target.value)}
        rows={4}
      />
    </div>
  );
}

export default function QuestionPanel({
  record,
  annotations,
  imageId,
  mode,
  nodeId,
  onAnswerChange,
  translationMap,
}) {
  const questions = mode === 'tree' ? treeQuestionsFor() : nodeQuestionsFor(record, nodeId);
  const answers = getAnswersBucket(annotations, imageId, mode, nodeId)?.answers || {};

  const headerTitle = mode === 'tree'
    ? '전체 트리 질문'
    : `${translatedLabel(imageId, nodeId, translationMap)} 노드 질문`;

  // const pills = mode === 'tree'
  //   ? [`image: ${imageId}`, 'mode: tree summary']
  //   : getInspectorPills(record, nodeId, translationMap);

  return (
    <section className="sectionCard">
      <div className="sectionHeaderWithMeta">
        <div>
          <h2 className="sectionTitle">{headerTitle}</h2>
          {/* <div className="statusPillsRow">
            {pills.map((pill) => (
              <span className="statusPill" key={pill}>{pill}</span>
            ))}
          </div> */}
        </div>
      </div>

      <div className="questionsGrid">
        {questions.map((question) => {
          const value = answers[question.id];
          const onChange = (nextValue) => onAnswerChange(mode, question.id, nextValue, nodeId);

          if (question.type === 'single_choice') {
            return <SingleChoiceQuestion key={question.id} question={question} value={value} onChange={onChange} />;
          }
          if (question.type === 'multi_choice') {
            return <MultiChoiceQuestion key={question.id} question={question} value={value} onChange={onChange} />;
          }
          return <TextQuestion key={question.id} question={question} value={value} onChange={onChange} />;
        })}
      </div>
    </section>
  );
}
