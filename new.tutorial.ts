import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import readline from "readline";

import * as dotenv from "dotenv";
import * as fs from "fs";
import * as path from "path";

dotenv.config();

const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
});
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-large",
});
const vectorStore = new MemoryVectorStore(embeddings);

// All files to process
const DOCS_DIR = "./docs";
let fileNames = [
  "meter-registration81e6694e-c12d-427c-b1dc-eeca44a778f5.pdf",
  "customer-data-and-agreements.pdf",
  "meter-works.pdf",
  "data-aggregation.pdf",
  "roi-market-message-guide-duos-and-transaction-payments-baseline-v5.0.pdf",
  "market-message-guide---data-processing-_v6-1.pdf",
  "roi-market-message-guide-unmetered-baseline-v5.0.pdf",
  "market-message-guide---market-gateway-activity-v6-1.pdf",
];

const files = fs
  .readdirSync(DOCS_DIR)
  .filter((file) => fs.statSync(path.join(DOCS_DIR, file)).isFile());

for (const file of files) {
  console.log(`\nProcessing file: ${file}`);

  const fileName = file;
  const filePath = path.join(DOCS_DIR, fileName);

  console.log(filePath);
  // Load and chunk contents of the PDF
  const pdfLoader = new PDFLoader(filePath);
  const docs = await pdfLoader.load();
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const allSplits = await splitter.splitDocuments(docs);

  // Index chunks
  await vectorStore.addDocuments(allSplits);
}

// Define prompt for question-answering
const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");

// Define state for application
const InputStateAnnotation = Annotation.Root({
  question: Annotation<string>,
});

const StateAnnotation = Annotation.Root({
  question: Annotation<string>,
  context: Annotation<Document[]>,
  answer: Annotation<string>,
});

// Define application steps
const retrieve = async (state: typeof InputStateAnnotation.State) => {
  const retrievedDocs = await vectorStore.similaritySearch(state.question);
  return { context: retrievedDocs };
};

const generate = async (state: typeof StateAnnotation.State) => {
  const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
  const messages = await promptTemplate.invoke({
    question: state.question,
    context: docsContent,
  });
  const response = await llm.invoke(messages);
  for (const doc of state.context) {
    console.info(`📄 Source: ${doc.metadata?.source}`);
  }
  return { answer: response.content };
};

// Compile application and test
const graph = new StateGraph(StateAnnotation)
  .addNode("retrieve", retrieve)
  .addNode("generate", generate)
  .addEdge("__start__", "retrieve")
  .addEdge("retrieve", "generate")
  .addEdge("generate", "__end__")
  .compile();

console.log();

// Setup readline interface for stdin interaction
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: "💬 Ask a question (or type 'exit' to quit): ",
});

console.log("🧠 Ready to answer questions about your document.");
rl.prompt();

rl.on("line", async (line) => {
  const question = line.trim();

  if (question.toLowerCase() === "exit") {
    rl.close();
    return;
  }

  try {
    const result = await graph.invoke({ question });
    console.log(`🗣️  ${result.answer}`);
  } catch (err) {
    console.error("❌ Error handling question:", err);
  }

  rl.prompt();
});

rl.on("close", () => {
  console.log("👋 Goodbye!");
  process.exit(0);
});
