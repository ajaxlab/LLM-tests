import pdf from 'pdf-parse';
import { readFile } from 'fs/promises';

export interface PdfParseResult {
  text: string;
  numPages: number;
  info: Record<string, unknown>;
}

/**
 * PDF 파일을 읽어서 텍스트로 변환합니다.
 * @param filePath PDF 파일 경로
 * @returns 변환된 텍스트와 메타데이터
 */
export async function pdfToText(filePath: string): Promise<PdfParseResult> {
  const buffer = await readFile(filePath);
  const data = await pdf(buffer);

  return {
    text: data.text,
    numPages: data.numpages,
    info: data.info,
  };
}

/**
 * PDF Buffer를 텍스트로 변환합니다.
 * @param buffer PDF 파일 버퍼
 * @returns 변환된 텍스트와 메타데이터
 */
export async function pdfBufferToText(
  buffer: Buffer
): Promise<PdfParseResult> {
  const data = await pdf(buffer);

  return {
    text: data.text,
    numPages: data.numpages,
    info: data.info,
  };
}
