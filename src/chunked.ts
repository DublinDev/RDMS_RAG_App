import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import * as fs from "fs";
import * as path from "path";

const docsDir = "./docs"; // Folder where all PDFs are located

async function splitAndTagPDF() {
  const files = fs.readdirSync(docsDir).filter((file) => file.endsWith(".pdf"));

  for (const file of files) {
    console.log(`\n📄 Processing: ${file}`);
    const pdfPath = path.join(docsDir, file);

    const loader = new PDFLoader(pdfPath);
    const pages = await loader.load();

    const fullText = pages.map((page) => page.pageContent).join("\n");

    const messagePattern = /(Message \d+:.*?)(?=Message \d+:|$)/gs;
    const matches = Array.from(fullText.matchAll(messagePattern));
    const hasRealMatches = matches.some((match) =>
      match[0].includes("Message ")
    );

    let documents: Document[] = [];

    if (hasRealMatches) {
      console.log("✅ Using 'Message XXX:' pattern");
      for (const match of matches) {
        const chunkText = match[1].trim();
        const headerMatch = chunkText.match(/Message (\d+):\s*(.*)/);

        const messageCode = headerMatch?.[1] || "unknown";
        const messageName = headerMatch?.[2] || "unknown";

        const doc = new Document({
          pageContent: chunkText,
          metadata: {
            message_code: messageCode,
            message_name: messageName,
            source_file: file,
          },
        });

        documents.push(doc);
      }
    } else {
      console.log(
        "✅ Falling back to 'Section Number Message Code Title' pattern"
      );
      const fallbackPattern =
        /(\d+\.\d+)\s+(\d{3})\s+([A-Za-z].*?)(?=\n\d+\.\d+|$)/gs;
      const fallbackMatches = Array.from(fullText.matchAll(fallbackPattern));

      if (fallbackMatches.length > 0) {
        for (const match of fallbackMatches) {
          const sectionNumber = match[1];
          const messageCode = match[2];
          const messageName = match[3].trim();

          const doc = new Document({
            pageContent: `${sectionNumber} ${messageCode} ${messageName}`,
            metadata: {
              message_code: messageCode,
              message_name: messageName,
              source_file: file,
            },
          });

          documents.push(doc);
        }
      } else {
        console.log(
          "✅ Falling back to 'Section Number Segment Title' pattern"
        );
        const segmentPattern = /(\d+\.\d+)\s+([A-Za-z][A-Za-z\s\/&]+)/gs;
        const segmentMatches = Array.from(fullText.matchAll(segmentPattern));

        for (const match of segmentMatches) {
          const sectionNumber = match[1];
          const segmentName = match[2].trim();

          const doc = new Document({
            pageContent: `${sectionNumber} ${segmentName}`,
            metadata: {
              section_number: sectionNumber,
              segment_name: segmentName,
              source_file: file,
            },
          });

          documents.push(doc);
        }
      }
    }

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 100,
    });

    const finalDocs: Document[] = [];

    for (const doc of documents) {
      const splits = await splitter.splitText(doc.pageContent);
      for (let i = 0; i < splits.length; i++) {
        finalDocs.push(
          new Document({
            pageContent: splits[i],
            metadata: {
              ...doc.metadata,
              split_id: i,
            },
          })
        );
      }
    }

    const outputFileName = path.basename(file, ".pdf") + ".jsonl";
    const outputPath = path.join("./mnt/data", outputFileName);
    const stream = fs.createWriteStream(outputPath, { flags: "w" });

    for (const doc of finalDocs) {
      const record = { text: doc.pageContent, metadata: doc.metadata };
      stream.write(JSON.stringify(record) + "\n");
    }

    stream.end();

    console.log(`✅ Split ${finalDocs.length} chunks, saved to ${outputPath}`);
  }
}

splitAndTagPDF();
