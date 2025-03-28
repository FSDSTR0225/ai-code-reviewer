import { readFileSync } from "fs";
import Anthropic from "@anthropic-ai/sdk";
import * as core from "@actions/core";
import { Octokit } from "@octokit/rest";
import parseDiff, { Chunk, File } from "parse-diff";
import { minimatch } from "minimatch";
import { exec } from "child_process";
import { promisify } from "util";
import { glob } from "glob";

const GITHUB_TOKEN: string = core.getInput("GITHUB_TOKEN");
const ANTHROPIC_API_KEY: string = core.getInput("ANTHROPIC_API_KEY");
const CLAUDE_MODEL: string = core.getInput("CLAUDE_MODEL");

const execAsync = promisify(exec);

const octokit = new Octokit({ auth: GITHUB_TOKEN });

const anthropic = new Anthropic({
  apiKey: ANTHROPIC_API_KEY,
});

interface PRDetails {
  owner: string;
  repo: string;
  pull_number: number;
  title: string;
  description: string;
}

async function addPullRequestComment(
  owner: string,
  repo: string,
  pull_number: number,
  body: string
): Promise<void> {
  await octokit.issues.createComment({
    owner,
    repo,
    issue_number: pull_number,
    body,
  });
}

// Get pylint score on python files
async function getPylintScore(): Promise<number> {
  const files: string[] = await glob("**/*.py", {
    ignore: ["venv/**", "env/**", "node_modules/**"],
  });
  console.log(files);
  if (files.length === 0) {
    console.log("No Python files found in the repository.");
    return 10;
  }

  const { stdout, stderr } = await execAsync(
    `pylint ${files.join(" ")} --exit-zero  --output-format=text`
  );

  if (stderr) {
    console.error("Pylint error:", stderr);
  }

  return parsePylint(stdout);
}

function parsePylint(pylintOutput: string): number {
  const match = pylintOutput.match(/Your code has been rated at (\d+\.\d+)/);
  return match ? parseFloat(match[1]) : 0;
}

async function getPRDetails(): Promise<PRDetails> {
  const { repository, number } = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH || "", "utf8")
  );
  const prResponse = await octokit.pulls.get({
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
  });
  return {
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
    title: prResponse.data.title ?? "",
    description: prResponse.data.body ?? "",
  };
}

async function getDiff(
  owner: string,
  repo: string,
  pull_number: number
): Promise<string | null> {
  const response = await octokit.pulls.get({
    owner,
    repo,
    pull_number,
    mediaType: { format: "diff" },
  });
  // @ts-expect-error - response.data is a string
  return response.data;
}

async function analyzeCode(
  parsedDiff: File[],
  prDetails: PRDetails
): Promise<Array<{ body: string; path: string; line: number }>> {
  const comments: Array<{ body: string; path: string; line: number }> = [];

  for (const file of parsedDiff) {
    if (file.to === "/dev/null") continue; // Ignore deleted files
    for (const chunk of file.chunks) {
      const prompt = createPrompt(file, chunk, prDetails);
      const aiResponse = await getAIResponse(prompt);
      if (aiResponse) {
        const newComments = createComment(file, chunk, aiResponse);
        if (newComments) {
          comments.push(...newComments);
        }
      }
    }
  }
  return comments;
}

function createPrompt(file: File, chunk: Chunk, prDetails: PRDetails): string {
  return `Eres un experto revisor de código full stack especializado en el stack MERN (MongoDB, Express, React, Node.js), con un profundo conocimiento de principios de ingeniería de software y mejores prácticas en desarrollo web. Tu tarea es revisar fragmentos de código (tanto frontend como backend) y proporcionar comentarios constructivos para mejorar su calidad. Sigue estas pautas durante tu revisión:

Legibilidad y simplicidad:
Evalúa la legibilidad general del código (JSX/JS/TS).

Identifica secciones innecesariamente complejas y sugiere simplificaciones.

Evita lógica innecesaria en los componentes de React o controladores de Express.

Convenciones de nombres y consistencia:
Verifica que los nombres de variables, funciones, componentes y clases sean descriptivos y sigan las convenciones de JavaScript/TypeScript (camelCase para variables/funciones, PascalCase para componentes).

Asegúrate de que haya consistencia en nombres de rutas, endpoints, componentes y archivos.

Estructura del código y modularidad:
Evalúa la organización del código en controladores, rutas, servicios y componentes.

Sugiere una mejor separación de responsabilidades si hay lógica mezclada (por ejemplo, lógica de negocio en rutas o lógica de presentación en componentes contenedores).

Revisa que se usen hooks personalizados y middlewares cuando sea apropiado.

Documentación y comentarios:
Verifica la presencia y calidad de documentación en funciones, clases y componentes si es necesario (por ejemplo, JSDoc o comentarios explicativos mínimos).

Evalúa si el código es autoexplicativo y evita comentarios innecesarios o redundantes.

Manejo de errores y validación de datos:
Verifica que haya manejo adecuado de errores en funciones asincrónicas y llamadas a la base de datos (por ejemplo, uso de try/catch o middlewares de error en Express).

Sugiere validaciones de entrada en formularios (React) y en el backend (por ejemplo, Joi, express-validator o lógica propia).

Eficiencia y rendimiento:
Identifica llamadas a la base de datos innecesarias o no optimizadas.

Sugiere el uso de useMemo, useCallback, paginación o lazy loading cuando sea apropiado.

Evalúa el tamaño de los componentes de React y si se pueden dividir.

Buenas prácticas y convenciones de desarrollo web:
Verifica que se sigan las guías de estilo (por ejemplo, ESLint, Prettier, convenciones REST para API).

Revisa el uso de características modernas del lenguaje (async/await, destructuring, optional chaining, etc.).

En el frontend, evalúa el uso de props, estado y hooks correctamente.

Tipado y anotaciones (opcional si usa TypeScript):
Sugiere el uso de tipado estático para mejorar la claridad del código y prevenir errores.

Verifica que se usen tipos correctamente en funciones, props, respuestas de API, etc.

Testabilidad:
Evalúa si el código es fácilmente testeable (por ejemplo, funciones puras, lógica desacoplada).

Sugiere patrones para facilitar pruebas unitarias o de integración (Jest, React Testing Library, Supertest, etc.).

Errores potenciales y casos límite:
Identifica errores lógicos, condiciones no controladas o posibles problemas en estados, props, o peticiones HTTP.

Señala casos límite que podrían no estar bien manejados (datos vacíos, errores de red, valores nulos, etc.).

Para cada problema identificado:
Explica claramente el problema y por qué es un problema.

Proporciona una sugerencia específica para mejorar el código.

Si corresponde, ofrece un fragmento de código que demuestre la mejora sugerida.

Instrucciones:
Proporciona la respuesta en el siguiente formato JSON:
{"reviews": [{"lineNumber": <número_de_línea>, "reviewComment": "<comentario_de_revisión>"}]}

No des comentarios positivos ni cumplidos.

Proporciona comentarios y sugerencias solo si hay algo que mejorar, de lo contrario, "reviews" debe ser un array vacío.

Escribe los comentarios en formato Markdown de GitHub.

Usa la descripción general solo como contexto y comenta únicamente el código.

IMPORTANTE: NUNCA sugieras añadir comentarios al código.

Revisa el siguiente diff de código en el archivo "${file.to}" y ten en cuenta el título y la descripción del pull request al redactar tu respuesta.

Pull request title: ${prDetails.title}
Pull request description:

---
${prDetails.description}
---

Git diff to review:

\`\`\`diff
${chunk.content}
${chunk.changes
  // @ts-expect-error - ln and ln2 exists where needed
  .map((c) => `${c.ln ? c.ln : c.ln2} ${c.content}`)
  .join("\n")}
\`\`\`
`;
}

async function getAIResponse(prompt: string): Promise<Array<{
  lineNumber: string;
  reviewComment: string;
}> | null> {
  const queryConfig = {
    model: CLAUDE_MODEL,
    temperature: 0.2,
    max_tokens: 700,
  };

  try {
    const response = await anthropic.messages.create({
      ...queryConfig,
      messages: [
        {
          role: "user",
          content: prompt,
        },
      ],
    });

    const content =
      response.content[0].type === "text" ? response.content[0].text : "";
    let parsedContent;
    try {
      parsedContent = JSON.parse(content);
    } catch (jsonError) {
      console.error("Error parsing initial JSON:", jsonError);
      const cleanedContent = content
        .replace(/[\u0000-\u001F\u007F-\u009F]/g, "")
        .replace(/\n/g, "\\n")
        .replace(/\r/g, "\\r")
        .replace(/\t/g, "\\t")
        .replace(/\f/g, "\\f")
        .replace(/"/g, '\\"')
        .replace(/\\'/g, "'");

      console.log("Cleaned content:", cleanedContent);

      try {
        parsedContent = JSON.parse(cleanedContent);
      } catch (secondJsonError) {
        console.error("Error parsing cleaned JSON:", secondJsonError);
        console.error("Cleaned content that failed to parse:", cleanedContent);
        return null;
      }
    }

    if (!parsedContent.reviews || !Array.isArray(parsedContent.reviews)) {
      console.error(
        "Parsed content does not contain a 'reviews' array:",
        parsedContent
      );
      return null;
    }

    // Ensure all reviewComments are properly escaped
    const sanitizedReviews = parsedContent.reviews.map(
      (review: { lineNumber: any; reviewComment: any }) => ({
        lineNumber: review.lineNumber,
        reviewComment: JSON.parse(JSON.stringify(review.reviewComment)),
      })
    );

    return sanitizedReviews;
  } catch (error) {
    console.error("Error in API call or processing:", error);
    if (error instanceof Error) {
      console.error("Error message:", error.message);
      console.error("Error stack:", error.stack);
    }
    return null;
  }
}

function createComment(
  file: File,
  chunk: Chunk,
  aiResponses: Array<{
    lineNumber: string;
    reviewComment: string;
  }>
): Array<{ body: string; path: string; line: number }> {
  return aiResponses.flatMap((aiResponse) => {
    if (!file.to) {
      return [];
    }

    return {
      body: aiResponse.reviewComment,
      path: file.to,
      line: Number(aiResponse.lineNumber),
    };
  });
}

async function createReviewComment(
  owner: string,
  repo: string,
  pull_number: number,
  comments: Array<{ body: string; path: string; line: number }>
): Promise<void> {
  await octokit.pulls.createReview({
    owner,
    repo,
    pull_number,
    comments,
    event: "COMMENT",
  });
}

async function main() {
  const prDetails = await getPRDetails();
  let diff: string | null;
  const eventData = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH ?? "", "utf8")
  );

  if (eventData.action === "opened") {
    diff = await getDiff(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number
    );
  } else if (eventData.action === "synchronize") {
    const newBaseSha = eventData.before;
    const newHeadSha = eventData.after;

    const response = await octokit.repos.compareCommits({
      headers: {
        accept: "application/vnd.github.v3.diff",
      },
      owner: prDetails.owner,
      repo: prDetails.repo,
      base: newBaseSha,
      head: newHeadSha,
    });

    diff = String(response.data);
  } else {
    console.log("Unsupported event:", process.env.GITHUB_EVENT_NAME);
    return;
  }

  if (!diff) {
    console.log("No diff found");
    return;
  }

  const parsedDiff = parseDiff(diff);

  const excludePatterns = core
    .getInput("exclude")
    .split(",")
    .map((s) => s.trim());

  const filteredDiff = parsedDiff.filter((file) => {
    return !excludePatterns.some((pattern) =>
      minimatch(file.to ?? "", pattern)
    );
  });

  const comments = await analyzeCode(filteredDiff, prDetails);
  const pylintScore = await getPylintScore();

  await addPullRequestComment(
    prDetails.owner,
    prDetails.repo,
    prDetails.pull_number,
    `The pylint score for this pull request is: ${pylintScore.toFixed(2)}/10`
  );

  if (comments.length > 0) {
    await createReviewComment(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number,
      comments
    );
  }
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
