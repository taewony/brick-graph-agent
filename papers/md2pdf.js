#!/usr/bin/env node
/**
 * md2pdf.js — Markdown → KIISE 논문지(JOK) 제출용 PDF 변환기
 *
 * 서식 기준 (papers/nhgcmdbskkrv.pdf = KIISE LaTeX 템플릿):
 *   - 용지: A4 (210 x 297 mm)
 *   - 여백: 위 30mm, 아래 20mm, 좌/우 10mm
 *   - 폰트: 한글 Nanum Myeongjo(명조), 코드 Courier
 *   - 크기: 제목 16pt, 영문제목 14pt, 절 제목 12pt, 본문 10pt, 표/참고문헌 9pt
 *
 * 사용법: node md2pdf.js <input.md> [output.pdf]
 */
const fs = require('fs');
const path = require('path');
const PDFDocument = require('pdfkit');
const MarkdownIt = require('markdown-it');

// ---------------------------------------------------------------------------
// 설정
// ---------------------------------------------------------------------------
const MM = 72 / 25.4;
const PAGE_W = 210, PAGE_H = 297;
const M_TOP = 30, M_BOTTOM = 20, M_LEFT = 10, M_RIGHT = 10;
const CONTENT_W_MM = PAGE_W - M_LEFT - M_RIGHT;

const FONT_BODY = 'C:/Windows/Fonts/NanumMyeongjo.ttf';
const FONT_BOLD = 'C:/Windows/Fonts/NanumMyeongjoBold.ttf';

const SIZE_TITLE = 16;
const SIZE_EN_TITLE = 14;
const SIZE_H2 = 12;
const SIZE_H3 = 11;
const SIZE_BODY = 10;
const SIZE_SMALL = 9;

// ---------------------------------------------------------------------------
const input = process.argv[2];
if (!input) { console.error('usage: node md2pdf.js <input.md> [output.pdf]'); process.exit(1); }
const output = process.argv[3] || input.replace(/\.md$/i, '.pdf');
const mdText = fs.readFileSync(input, 'utf8');
const md = new MarkdownIt({ html: false, linkify: true, breaks: false, typographer: false });
const tokens = md.parse(mdText, {});

// ---------------------------------------------------------------------------
const doc = new PDFDocument({
  size: 'A4', bufferPages: true,
  margins: { top: M_TOP * MM, bottom: M_BOTTOM * MM, left: M_LEFT * MM, right: M_RIGHT * MM },
  info: { Title: path.basename(input, path.extname(input)), Creator: 'md2pdf (pdfkit)' }
});
doc.registerFont('Body', FONT_BODY);
doc.registerFont('Bold', FONT_BOLD);
doc.pipe(fs.createWriteStream(output));

const leftX = M_LEFT * MM;
const contentW = CONTENT_W_MM * MM;
const pageWpt = PAGE_W * MM;
const pageHpt = PAGE_H * MM;

// ---------------------------------------------------------------------------
// 인라인
// ---------------------------------------------------------------------------
function flattenInline(tokens) {
  const out = [];
  const stack = [{ font: 'Body' }];
  for (const t of tokens || []) {
    const cur = stack[stack.length - 1];
    switch (t.type) {
      case 'text': out.push({ text: t.content, font: cur.font }); break;
      case 'code_inline': out.push({ text: t.content, font: 'Code' }); break;
      case 'strong_open': stack.push({ font: 'Bold' }); break;
      case 'strong_close': stack.pop(); break;
      case 'em_open': stack.push({ font: 'Bold' }); break;
      case 'em_close': stack.pop(); break;
      case 'softbreak': out.push({ text: ' ', font: cur.font }); break;
      case 'hardbreak': out.push({ text: '\n', font: cur.font }); break;
      default: break;
    }
  }
  return out;
}
function fontName(f) { return f === 'Code' ? 'Courier' : (f === 'Bold' ? 'Bold' : 'Body'); }
function inlineText(tokens) { return flattenInline(tokens).map(s => s.text).join(''); }

function renderSegments(segs, size, opts) {
  opts = opts || {};
  let started = false;
  for (const seg of segs) {
    if (seg.text === '\n') { if (started) doc.text('', { continued: false }); started = false; continue; }
    doc.font(fontName(seg.font)).fontSize(size);
    if (!started) {
      const o = {
        continued: true,
        align: opts.align || 'justify',
        width: opts.width !== undefined ? opts.width : contentW,
        lineGap: opts.lineGap !== undefined ? opts.lineGap : 2
      };
      if (opts.x !== undefined) { o.x = opts.x; o.y = opts.y; }
      doc.text(seg.text, o);
      started = true;
    } else {
      doc.text(seg.text, { continued: true });
    }
  }
  if (started) doc.text('', { continued: false });
}

// ---------------------------------------------------------------------------
// 블록
// ---------------------------------------------------------------------------
function renderTable(headerRows, bodyRows) {
  const ncols = headerRows[0] ? headerRows[0].length : (bodyRows[0] ? bodyRows[0].length : 1);
  const rows = [...headerRows, ...bodyRows];
  const pad = 3;
  const colW = contentW / ncols;
  const cellW = colW - pad * 2;

  const rowHeights = rows.map(r => {
    let maxH = 0;
    for (const cell of r) {
      const h = doc.font('Body').fontSize(SIZE_SMALL).heightOfString(cell || '', { width: cellW, lineGap: 1 });
      if (h + pad * 2 > maxH) maxH = h + pad * 2;
    }
    return maxH;
  });

  for (let r = 0; r < rows.length; r++) {
    const rowH = rowHeights[r];
    if (doc.y + rowH > pageHpt - M_BOTTOM * MM) doc.addPage();
    const y0 = doc.y;
    if (r < headerRows.length) {
      doc.save().fillColor('#f0f0f0').rect(leftX, y0, contentW, rowH).fill().restore();
    }
    for (let c = 0; c < ncols; c++) {
      const x = leftX + c * colW;
      const isHeader = r < headerRows.length;
      doc.font(isHeader ? 'Bold' : 'Body').fontSize(SIZE_SMALL)
        .text(rows[r][c] || '', x + pad, y0 + pad, { width: cellW, lineGap: 1 });
    }
    doc.save().lineWidth(0.5).strokeColor('#999999');
    doc.rect(leftX, y0, contentW, rowH).stroke();
    for (let c = 1; c < ncols; c++) doc.moveTo(leftX + c * colW, y0).lineTo(leftX + c * colW, y0 + rowH).stroke();
    doc.restore();
    doc.y = y0 + rowH;
  }
  doc.moveDown(0.5);
}

function renderCodeBlock(content) {
  const lines = content.replace(/\n$/, '').split('\n');
  const pad = 5, fs = 8.5, lineH = fs * 1.35;
  const boxH = lines.length * lineH + pad * 2;
  if (doc.y + boxH > pageHpt - M_BOTTOM * MM && doc.y > M_TOP * MM + 10) doc.addPage();
  const y0 = doc.y;
  doc.save().fillColor('#f7f7f7').rect(leftX, y0, contentW, boxH).fill().restore();
  doc.save().fillColor('#111111').font('Courier').fontSize(fs);
  for (let i = 0; i < lines.length; i++) {
    doc.text(lines[i] || ' ', leftX + pad, y0 + pad + i * lineH, { width: contentW - pad * 2, lineBreak: false });
  }
  doc.restore();
  doc.y = y0 + boxH;
  doc.moveDown(0.5);
}

function drawHr() {
  const y = doc.y + 3;
  doc.save().moveTo(leftX, y).lineTo(leftX + contentW, y).lineWidth(0.6).strokeColor('#444444').stroke().restore();
  doc.y = y + 8;
}

function heading(text, size, opts) {
  opts = opts || {};
  doc.font('Bold').fontSize(size);
  doc.text(text, leftX, doc.y, {
    align: opts.align || 'left', width: contentW, lineGap: 1
  });
  doc.moveDown((opts.after !== undefined ? opts.after : size / SIZE_BODY * 0.5));
}

function abstractLabel(raw) {
  if (/국문 요약|요약/.test(raw)) return '요 약';
  if (/Abstract/.test(raw)) return 'Abstract';
  if (/국문 키워드/.test(raw)) return '키워드';
  if (/영문 키워드|Keywords/.test(raw)) return 'Keywords';
  return raw.replace(' (Abstract)', '');
}

// ---------------------------------------------------------------------------
let i = 0;
let inAbstract = true;
let pendingKeywords = null; // 'ko' | 'en'

while (i < tokens.length) {
  const t = tokens[i];

  if (t.type === 'hr') { drawHr(); inAbstract = false; i++; continue; }

  if (t.type === 'heading_open') {
    const level = parseInt(t.tag.slice(1), 10);
    const text = inlineText((tokens[i + 1] || {}).children || []);
    i += 3;

    if (inAbstract && level >= 2) {
      if (level === 2) heading(text, SIZE_EN_TITLE, { align: 'center', after: 0.3 });
      else {
        if (/키워드|Keywords/.test(text)) {
          pendingKeywords = /Keywords/.test(text) ? 'en' : 'ko';
        } else {
          heading(abstractLabel(text), SIZE_H3, { align: 'center', after: 0.2 });
        }
      }
    } else if (level === 1) {
      heading(text, SIZE_TITLE, { align: 'center', after: 0.3 });
    } else if (level === 2) {
      doc.moveDown(0.6);
      heading(text, SIZE_H2, { align: 'left', after: 0.5 });
    } else if (level === 3) {
      doc.moveDown(0.4);
      heading(text, SIZE_H3, { align: 'left', after: 0.4 });
    } else {
      doc.moveDown(0.3);
      heading(text, SIZE_BODY, { align: 'left', after: 0.3 });
    }
    continue;
  }

  if (t.type === 'paragraph_open') {
    const segs = flattenInline((tokens[i + 1] || {}).children || []);
    i += 3;
    if (segs.length === 0) continue;
    if (pendingKeywords) {
      const prefix = pendingKeywords === 'en' ? 'Keywords: ' : '키워드: ';
      renderSegments([{ text: prefix, font: 'Bold' }, ...segs], SIZE_BODY, { align: 'left', lineGap: 2 });
      pendingKeywords = null;
    } else {
      renderSegments(segs, SIZE_BODY, { align: 'justify', lineGap: 2 });
    }
    doc.moveDown(0.4);
    continue;
  }

  if (t.type === 'blockquote_open') {
    const parts = [];
    i++;
    while (i < tokens.length && tokens[i].type !== 'blockquote_close') {
      if (tokens[i].type === 'paragraph_open') {
        parts.push(inlineText((tokens[i + 1] || {}).children || []));
        i += 3;
      } else i++;
    }
    i++;
    doc.moveDown(0.2);
    doc.font('Bold').fontSize(SIZE_BODY).text(parts.join(' '), leftX, doc.y, { align: 'center', width: contentW, lineGap: 1 });
    doc.moveDown(0.6);
    continue;
  }

  if (t.type === 'bullet_list_open') {
    i++;
    while (i < tokens.length && tokens[i].type !== 'bullet_list_close') {
      if (tokens[i].type === 'list_item_open') {
        i++;
        let text = '';
        while (i < tokens.length && tokens[i].type !== 'list_item_close') {
          if (tokens[i].type === 'paragraph_open') { text += inlineText((tokens[i + 1] || {}).children || []); i += 3; }
          else i++;
        }
        i++;
        const segs = flattenInline(md.parseInline(text, {})[0].children || []);
        const indent = 12;
        const y = doc.y;
        doc.font('Body').fontSize(SIZE_BODY).text('•', leftX, y, { width: indent, lineBreak: false });
        renderSegments(segs, SIZE_BODY, { x: leftX + indent, y, width: contentW - indent, align: 'left', lineGap: 2 });
        doc.moveDown(0.15);
      } else i++;
    }
    i++;
    continue;
  }

  if (t.type === 'ordered_list_open') {
    i++;
    let num = 1;
    while (i < tokens.length && tokens[i].type !== 'ordered_list_close') {
      if (tokens[i].type === 'list_item_open') {
        i++;
        let text = '';
        while (i < tokens.length && tokens[i].type !== 'list_item_close') {
          if (tokens[i].type === 'paragraph_open') { text += inlineText((tokens[i + 1] || {}).children || []); i += 3; }
          else i++;
        }
        i++;
        const segs = flattenInline(md.parseInline(text, {})[0].children || []);
        const label = num + '.';
        const indent = 18;
        const y = doc.y;
        doc.font('Bold').fontSize(SIZE_SMALL).text(label, leftX, y, { width: indent, lineBreak: false });
        renderSegments(segs, SIZE_SMALL, { x: leftX + indent, y, width: contentW - indent, align: 'left', lineGap: 1 });
        doc.moveDown(0.15);
        num++;
      } else i++;
    }
    i++;
    continue;
  }

  if (t.type === 'fence' || t.type === 'code_block') { renderCodeBlock(t.content); i++; continue; }

  if (t.type === 'table_open') {
    i++;
    const headerRows = [], bodyRows = [];
    let inHead = false, inBody = false, curRow = null;
    while (i < tokens.length && tokens[i].type !== 'table_close') {
      const tt = tokens[i];
      if (tt.type === 'thead_open') { inHead = true; i++; continue; }
      if (tt.type === 'thead_close') { inHead = false; i++; continue; }
      if (tt.type === 'tbody_open') { inBody = true; i++; continue; }
      if (tt.type === 'tbody_close') { inBody = false; i++; continue; }
      if (tt.type === 'tr_open') { curRow = []; i++; continue; }
      if (tt.type === 'tr_close') { if (curRow) (inHead ? headerRows : bodyRows).push(curRow); curRow = null; i++; continue; }
      if (tt.type === 'th_open' || tt.type === 'td_open') {
        const cell = inlineText((tokens[i + 1] || {}).children || []);
        if (curRow) curRow.push(cell);
        i += 3; continue;
      }
      i++;
    }
    i++;
    renderTable(headerRows, bodyRows);
    continue;
  }

  i++;
}

// ---------------------------------------------------------------------------
// 페이지 번호
// ---------------------------------------------------------------------------
const range = doc.bufferedPageRange();
for (let p = range.start; p < range.start + range.count; p++) {
  doc.switchToPage(p);
  doc.font('Body').fontSize(9).fillColor('#000000');
  doc.text(String(p - range.start + 1), 0, pageHpt - (M_BOTTOM * MM) / 2 - 5, {
    width: pageWpt, align: 'center', lineBreak: false
  });
}

doc.end();
